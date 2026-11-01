"""Candidate generation (union of sources) + per (user, show) features for the re-ranker."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

import methods
from protocol import load

TOPN = {"ppr": 30, "als": 30, "rules": 30, "pop": 20}


def rank_rows(S):
    """0 = best, per row."""
    r = np.empty_like(S, dtype=np.int32)
    o = np.argsort(-S, axis=1, kind="stable")
    np.put_along_axis(r, o, np.arange(S.shape[1], dtype=np.int32)[None].repeat(S.shape[0], 0), axis=1)
    return r


def sub_counts(F, M, h):
    c = np.searchsorted(M.shows, h.tv_show_id.values)
    f = (h.frac >= 0.8).values
    return sp.csr_matrix((np.ones(f.sum()), (h.uid.values[f], c[f])), shape=M.c80.shape)


def profile(keys, vals, weights, n, nbin):
    P = sp.csr_matrix((weights, (keys, vals)), shape=(n, nbin)).toarray()
    return P / np.maximum(P.sum(1, keepdims=True), 1e-12)


def build(F, users, best):
    """Returns the candidate pair table with features (+ label when the fold has truth) and candidate recall."""
    M = methods.Mat(F)
    cands, nw, nwt = F.cands, F.n_weeks, F.n_tgt_weeks
    cols = M.cand_cols
    sc = {"ppr": methods.ppr(F, M, users, **best["ppr"]),
          "als": methods.als(F, M, users, **best["als"]),
          "rules": methods.rules(F, M, users, **best["rules"])[0]}
    sec = M.sec[users][:, cols].toarray()
    mask = sec > 0
    rk = {}
    for k, S in sc.items():
        rk[k] = rank_rows(S)
        mask |= (rk[k] < TOPN[k]) & (S > 0)
    pos = {c: i for i, c in enumerate(cands)}
    pop_cols = [pos[p] for p in F.pop[:TOPN["pop"]]]
    mask[:, pop_cols] = True
    ui, ci = np.nonzero(mask)
    u = users[ui]
    d = pd.DataFrame({"uid": u.astype(np.int32), "tv_show_id": cands[ci]})

    # history of the user with the show
    c80 = M.c80[users][:, cols]
    d["c80_rate"] = np.asarray(c80[ui, ci]).ravel() / nw
    d["sec_rate"] = sec[ui, ci] / nw
    h = F.hist
    for wk in (4, 2):
        cw = sub_counts(F, M, h[h.week > F.h1 - wk])[users][:, cols]
        d[f"c80_last{wk}"] = np.asarray(cw[ui, ci]).ravel()
    cw2 = cw.toarray()
    top2 = rank_rows(cw2 + 1e-9 * sec) < 5
    d["top5_last2"] = (top2[ui, ci] & (cw2[ui, ci] > 0)).astype(np.int8)
    g = h[h.frac >= 0.8].assign(day=lambda x: x.start // 86400)
    agg = g.groupby(["uid", "tv_show_id"]).agg(n_days=("day", "nunique"), last_wk=("week", "max"))
    d = d.join(agg, on=["uid", "tv_show_id"])
    d["n_days"] = d.n_days.fillna(0) / nw
    d["weeks_since"] = (F.h1 - d.last_wk).fillna(99)
    d = d.drop(columns="last_wk")

    # show side
    sh = F.sched_hist.groupby("tv_show_id").size()
    st = F.sched_tgt.groupby("tv_show_id")
    d["air_hist_rate"] = d.tv_show_id.map(sh).fillna(0).values / nw
    d["air_tgt_rate"] = d.tv_show_id.map(st.size()).values / nwt
    d["share_watched"] = d.c80_rate / np.maximum(d.air_hist_rate, 1 / nw)
    d["expected"] = d.share_watched.clip(upper=1) * d.air_tgt_rate
    d["dur_med"] = d.tv_show_id.map(st.dur.median()).values
    meta = load()["sched"].groupby("tv_show_id").tv_show_category.first()
    d["category"] = d.tv_show_id.map(meta).astype("category").cat.codes
    p80 = h[h.frac >= 0.8]
    d["pop_rate"] = d.tv_show_id.map(p80.groupby("tv_show_id").uid.nunique()).fillna(0).values / nw
    d["pop_last4"] = d.tv_show_id.map(p80[p80.week > F.h1 - 4].groupby("tv_show_id").uid.nunique()).fillna(0).values

    # user-channel / time-profile affinity (views of the history window, incl. unlabelled airtime)
    v = F.views
    nu = len(load()["users"])
    vs = (v.stop - v.start).clip(lower=0).values.astype(float)
    main_ch = st.channel_id.agg(lambda x: x.value_counts().index[0])
    ch = d.tv_show_id.map(main_ch).values
    uc = v.assign(s=vs).groupby(["uid", "channel_id"]).s.sum()
    ut = uc.groupby(level=0).sum()
    d["chan_share"] = pd.Series(uc.reindex(list(zip(d.uid, ch))).values).fillna(0).values / \
        np.maximum(d.uid.map(ut).fillna(0).values, 1)
    hour = (v.start.values // 3600 + 3) % 24  # MSK hour, only used as a profile
    wd = ((v.start.values // 86400) + 3) % 7  # 1970-01-01 was a Thursday -> Monday = 0
    Uh, Uw = profile(v.uid.values, hour, vs, nu, 24), profile(v.uid.values, wd, vs, nu, 7)
    sg = F.sched_tgt
    sidx = np.searchsorted(cands, sg.tv_show_id.values)
    Sh = profile(sidx, (sg.start.values // 3600 + 3) % 24, np.ones(len(sg)), len(cands), 24)
    Sw = profile(sidx, ((sg.start.values // 86400) + 3) % 7, np.ones(len(sg)), len(cands), 7)
    d["hour_overlap"] = np.minimum(Uh[u], Sh[ci]).sum(1)
    d["wday_overlap"] = np.minimum(Uw[u], Sw[ci]).sum(1)

    # user side
    tot80 = np.asarray(M.c80[users].sum(1)).ravel()
    d["user_c80_rate"] = tot80[ui] / nw
    d["user_n_shows"] = np.asarray((M.c80[users] > 0).sum(1)).ravel()[ui]

    # method scores
    for k, S in sc.items():
        d[f"{k}_score"] = S[ui, ci]
        d[f"{k}_rank"] = rk[k][ui, ci]
    d["pop_rank"] = d.tv_show_id.map({p: i for i, p in enumerate(F.pop)}).values

    rec = None
    if F.truth:
        tr = set((uu, s) for uu in users for s in F.truth.get(uu, []))
        d["label"] = [int((a, b) in tr) for a, b in zip(d.uid.values, d.tv_show_id.values)]
        rec = d.label.sum() / max(len(tr), 1)
    return d, rec
