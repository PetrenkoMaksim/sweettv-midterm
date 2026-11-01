"""python src/run.py validate | final | examples   (needs cache/ from python src/prep.py)"""
import json
import warnings
import sys
import time

import numpy as np
import pandas as pd

import features
import methods
import ranker
from protocol import RES, Fold, fill, log_attempt, log_timing, popularity, topk, write_submission

warnings.filterwarnings("ignore", category=FutureWarning)
BEST = RES / "best_configs.json"
PPR_GRID = [dict(alpha=a, weight=w, restart=r) for w in ("logsec", "c80") for r in ("user", "items")
            for a in (0.5, 0.7, 0.85)]
ALS_GRID = [dict(factors=f, reg=r, conf="logsec", a=1.0) for f in (32, 64) for r in (0.01, 0.1)] + \
           [dict(factors=64, reg=0.1, conf="c80", a=1.0), dict(factors=64, reg=0.1, conf="logsec", a=5.0)]
RULES_GRID = [dict(min_sup=s, min_conf=c, agg=g, self_rule=True, metric="conf") for s in (5, 20) for c in (0.05, 0.2)
              for g in ("max", "sum")] + \
             [dict(min_sup=5, min_conf=0.05, agg="sum", self_rule=False, metric="conf"),
              dict(min_sup=5, min_conf=0.05, agg="sum", self_rule=True, metric="lift")]

def cfg_str(c):
    return " ".join(f"{k}={v}" for k, v in c.items())


def run_method(F, M, name, cfg, users, log=True):
    t = time.time()
    if name == "ppr":
        S = methods.ppr(F, M, users, **cfg)
    elif name == "als":
        S = methods.als(F, M, users, **cfg)
    elif name == "rules":
        S, n = methods.rules(F, M, users, **cfg)
    else:
        S = methods.repeat(F, M, users)
    preds = topk(S, F.cands, users, F.pop)
    secs = time.time() - t
    if log and F.truth:
        a, b = F.evaluate(preds)
        note = f"n_rules={n}" if name == "rules" else ""
        log_attempt(name, cfg_str(cfg), F.name, a, b, secs, note)
        log_timing(name, cfg_str(cfg), F.name, secs, len(users))
        return preds, a
    return preds, None


def validate():
    F = Fold("A")
    M = methods.Mat(F)
    users = F.users_all
    print(f"fold A: users with truth {len(users)}, test users among them {len(F.users_test)}, "
          f"candidates {len(F.cands)}, history shows {len(M.shows)}")
    for lw in (None, 4):
        t = time.time()
        pop = popularity(F.hist, F.cands, lw)
        a, b = F.evaluate({u: pop[:5] for u in users})
        log_attempt("popularity", f"last_weeks={lw or 'all'}", "A", a, b, time.time() - t)
    run_method(F, M, "repeat", {}, users)
    # PPR: closed form vs power iteration on 200 users
    u0 = users[:200]
    c0 = PPR_GRID[2]
    d = np.abs(methods.ppr(F, M, u0, **c0) - methods.ppr_power(M, u0, **c0)).max()
    (RES / "ppr_check.txt").write_text(f"{cfg_str(c0)}: max |closed form - 200 power iterations| over 200 users "
                                       f"x {len(F.cands)} candidates = {d:.2e}\n")
    print("ppr check", d)
    best = {}
    for name, grid in (("ppr", PPR_GRID), ("als", ALS_GRID), ("rules", RULES_GRID)):
        res = [(run_method(F, M, name, c, users)[1], i) for i, c in enumerate(grid)]
        best[name] = grid[max(res)[1]]
    BEST.write_text(json.dumps(best, indent=1))
    FB = Fold("B")
    MB = methods.Mat(FB)
    for name in ("popularity", "repeat", "ppr", "als", "rules"):
        if name == "popularity":
            a, b = FB.evaluate({u: FB.pop[:5] for u in FB.users_all})
            log_attempt("popularity", "last_weeks=all", "B", a, b, 0)
        else:
            run_method(FB, MB, name, best.get(name, {}), FB.users_all)
    hybrid_validate(best)


def hybrid_validate(best=None):
    best = best or json.loads(BEST.read_text())
    F = Fold("A")
    hist_users = np.array(sorted(set(F.hist.uid)))
    users = np.intersect1d(F.users_all, hist_users)
    t = time.time()
    d, rec = features.build(F, users, best)
    tb = time.time() - t
    print(f"candidates: {len(d)} rows, {len(d) / len(users):.1f} per user, recall of truth {rec:.4f} ({tb:.0f}s)")
    (RES / "candidate_recall.txt").write_text(
        f"fold A, users with history and truth: {len(users)}; candidate rows {len(d)} "
        f"({len(d) / len(users):.1f}/user); truth items inside candidates: {rec:.4f} "
        f"(truth of all users with truth incl. cold: {d.label.sum() / sum(len(v) for v in F.truth.values()):.4f})\n")
    d.to_parquet(RES.parent / "cache/hybrid_A.parquet")
    res = []
    for obj, seeds in (("binary", (42,)), ("lambdarank", (42,)), ("binary", (42, 43, 44))):
        t = time.time()
        p = ranker.oof(d, obj, seeds)
        preds = ranker.to_preds(d, p)
        preds = {u: fill(preds.get(u, []), F.pop) for u in F.users_all}
        a, b = F.evaluate(preds)
        log_attempt("hybrid", f"lgbm {obj} seeds={len(seeds)} 3-fold GroupKFold", "A", a, b, time.time() - t + tb)
        res.append((b, a, obj, len(seeds), preds))
    b, a, obj, ns, preds = max(res, key=lambda r: r[0])
    (RES / "hybrid_choice.json").write_text(json.dumps({"objective": obj, "n_seeds": ns, "map5_all": a, "map5_test": b}))
    pd.Series({int(k): " ".join(map(str, v)) for k, v in preds.items()}).to_csv(RES.parent / "cache/hybrid_A_oof.csv")
    m = ranker.fit(d, "binary")
    ranker.importance(m, d).to_csv(RES / "feature_importance.csv", header=["gain_share"])
    print(ranker.importance(m, d))


def baselines():
    """Quick safety net: popularity + repeat submissions for the real test window."""
    F = Fold("final")
    M = methods.Mat(F)
    write_submission({u: F.pop[:5] for u in F.users_all}, "popularity", F.pop)
    write_submission(run_method(F, M, "repeat", {}, F.users_all, log=False)[0], "repeat", F.pop)


def final():
    best = json.loads(BEST.read_text())
    ch = json.loads((RES / "hybrid_choice.json").read_text())
    obj, seeds = ch["objective"], (42, 43, 44)[:ch["n_seeds"]]
    F = Fold("final")
    M = methods.Mat(F)
    users = F.users_all  # all 1,260 test users
    if "singles" in sys.argv:
        write_submission({u: F.pop[:5] for u in users}, "popularity", F.pop)
        for name in ("repeat", "ppr", "als", "rules"):
            preds, _ = run_method(F, M, name, best.get(name, {}), users, log=False)
            write_submission(preds, name, F.pop)
    # hybrid: train on fold A (+ fold B) candidate tables, score the final candidates
    t = time.time()
    tr = [pd.read_parquet(RES.parent / "cache/hybrid_A.parquet")]
    FB = Fold("B")
    ub = np.intersect1d(FB.users_all, FB.hist.uid.unique())  # was a set() rebuilt per user (minutes)
    db = features.build(FB, ub, best)[0]
    db["uid"] += 10**6  # separate lambdarank queries for the same user in fold A and fold B
    tr.append(db)
    tr = pd.concat(tr, ignore_index=True)
    hu = np.intersect1d(users, F.hist.uid.unique())
    d, _ = features.build(F, hu, best)
    print(f"train rows {len(tr)}, predict rows {len(d)}, {time.time() - t:.0f}s", flush=True)
    p = np.mean([ranker.predict(ranker.fit(tr, obj, s), d) for s in seeds], axis=0)
    preds = ranker.to_preds(d, p)
    log_timing("hybrid", f"final fit A+B rows={len(tr)}, predict rows={len(d)}", "final", time.time() - t, len(hu))
    write_submission(preds, "hybrid", F.pop)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    {"validate": validate, "hybrid": hybrid_validate, "final": final, "baselines": baselines,
     "examples": lambda: __import__("examples").main()}[cmd]()
