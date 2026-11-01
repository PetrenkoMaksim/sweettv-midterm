"""Single methods. Each returns a dense score array (len(users), len(F.cands)); topk() turns it into a Top-5."""
import numpy as np
import scipy.sparse as sp

from protocol import load


class Mat:
    """User x show matrices of the history window (rows = uid, columns = shows of history U candidates)."""

    def __init__(self, F):
        h = F.hist
        self.shows = np.union1d(h.tv_show_id.unique(), F.cands)
        self.col = {s: i for i, s in enumerate(self.shows)}
        self.cand_cols = np.array([self.col[c] for c in F.cands])
        nu = len(load()["users"])
        c = np.searchsorted(self.shows, h.tv_show_id.values)
        shape = (nu, len(self.shows))
        self.sec = sp.csr_matrix((h.ov.values.astype(float), (h.uid.values, c)), shape=shape)
        f = (h.frac >= 0.8).values
        self.c80 = sp.csr_matrix((np.ones(f.sum()), (h.uid.values[f], c[f])), shape=shape)
        self.c80.sum_duplicates()
        self.sec.sum_duplicates()


def repeat(F, M, users):
    """Own >= 80 % airings count, tie-break by watched seconds."""
    c = M.c80[users][:, M.cand_cols].toarray()
    s = M.sec[users][:, M.cand_cols].toarray()
    return c + s / (s.max() + 1)


def graph(M, weight):
    W = M.c80.copy() if weight == "c80" else M.sec.copy()
    if weight == "logsec":
        W.data = np.log1p(W.data)
    return W.tocsr()


def rownorm(W):
    return (sp.diags(1 / np.maximum(np.asarray(W.sum(1)).ravel(), 1e-12)) @ W).tocsr()


def ppr(F, M, users, alpha=0.85, weight="logsec", restart="user"):
    """Personalized PageRank on the bipartite user-show graph (restart prob 1-alpha).
    The walk alternates user -> show -> user, so the show part of the stationary vector is
    (1-alpha) * q * (I - alpha^2 T)^-1 with T = P_show->user @ P_user->show (show x show) and
    q = alpha * P_user->show[u] (restart at the user) or q = the user's normalised show weights.
    This is the limit of the power iteration (checked against ppr_power in results/ppr_check.txt)."""
    W = graph(M, weight)
    Pui, Piu = rownorm(W), rownorm(W.T.tocsr())
    T = (Piu @ Pui).toarray()
    K = np.linalg.inv(np.eye(T.shape[0]) - alpha**2 * T)[:, M.cand_cols]
    # restart on own shows uses uniform weights (edge-weighted restart would equal alpha * P_user->show[u])
    q = (alpha * Pui[users] if restart == "user" else rownorm((W[users] > 0).astype(float))).toarray()
    return ((1 - alpha) * q @ K).astype(np.float32)


def ppr_power(M, users, alpha=0.85, weight="logsec", restart="user", iters=200):
    """Plain power iteration over all user+show nodes (slow; only used to check ppr())."""
    W = graph(M, weight)
    nu, ni = W.shape
    A = sp.bmat([[None, W], [W.T, None]]).tocsr()
    PT = rownorm(A).T.tocsr()
    E = np.zeros((nu + ni, len(users)))
    if restart == "user":
        E[users, np.arange(len(users))] = 1
    else:
        E[nu:] = rownorm((W[users] > 0).astype(float)).toarray().T
    X = E.copy()
    for _ in range(iters):
        X = (1 - alpha) * E + alpha * (PT @ X)
    return X[nu + M.cand_cols].T


def als(F, M, users, factors=64, reg=0.1, a=1.0, conf="logsec", iters=15):
    """Implicit ALS (Hu et al. 2008) via `implicit`; confidence = 1 + a*log(1+sec/60) or 1 + a*count80."""
    from implicit.als import AlternatingLeastSquares
    if conf == "logsec":
        C = M.sec.copy()
        C.data = 1 + a * np.log1p(C.data / 60)
    else:
        C = M.c80.copy()
        C.data = 1 + a * C.data
    m = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, random_state=42)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, "blas"):
        m.fit(C.tocsr(), show_progress=False)
    U, V = np.asarray(m.user_factors), np.asarray(m.item_factors)
    return (U[users] @ V[M.cand_cols].T).astype(np.float32)


def rules(F, M, users, min_sup=10, min_conf=0.05, metric="conf", agg="sum", self_rule=True):
    """Pairwise Apriori rules a -> b over baskets = shows with >= 1 airing watched >= 80 %."""
    B = (M.c80 > 0).astype(float).tocsr()
    sup = np.asarray(B.sum(0)).ravel()
    freq = sup >= min_sup  # frequent 1-itemsets (downward closure: pairs only among them)
    B = B[:, freq]
    idx = np.where(freq)[0]
    co = (B.T @ B).toarray()
    s1 = sup[idx]
    conf = co / s1[:, None]
    np.fill_diagonal(conf, 0)
    ok = (co >= min_sup) & (conf >= min_conf)
    n_users = (np.asarray(M.c80.sum(1)).ravel() > 0).sum()
    val = conf if metric == "conf" else conf / (s1[None, :] / n_users)
    R = np.where(ok, val, 0.0)
    n_rules = int(ok.sum() - ok.diagonal().sum())
    if self_rule:
        np.fill_diagonal(R, R.max() + 1 if metric == "lift" else 1.0)
    Ub = B[users].toarray()
    if agg == "sum":
        S = Ub @ R
    else:
        S = np.zeros((len(users), len(idx)))
        for i in range(len(users)):
            nz = np.flatnonzero(Ub[i])
            if len(nz):
                S[i] = R[nz].max(0)
    full = np.zeros((len(users), len(M.shows)), dtype=np.float32)
    full[:, idx] = S
    return full[:, M.cand_cols], n_rules
