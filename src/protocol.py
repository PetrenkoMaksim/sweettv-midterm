"""Folds, truth (competition rule), candidate pools, MAP@5, popularity fill, submission writer, attempt log."""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIT_ROOT", Path(__file__).resolve().parents[1]))
CACHE, RES, SUB, DATA = ROOT / "cache", ROOT / "results", ROOT / "submissions", ROOT / "data"
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import apk, make_truth, mapk  # noqa: E402

FOLDS = {"A": (11, 20, 21, 30), "B": (11, 24, 25, 30), "final": (11, 30, 31, 42)}
_cache = {}


def load():
    if not _cache:
        _cache["inter"] = pd.read_parquet(CACHE / "interactions.parquet")
        _cache["sched"] = pd.read_parquet(CACHE / "schedule.parquet")
        _cache["users"] = pd.read_parquet(CACHE / "users.parquet")
        _cache["views"] = pd.read_parquet(CACHE / "views.parquet")
    return _cache


class Fold:
    """History interactions, target-window schedule (candidate pool) and truth for one split."""

    def __init__(self, name):
        d = load()
        h0, h1, t0, t1 = FOLDS[name]
        it, sc = d["inter"], d["sched"]
        self.name, self.h1, self.n_weeks = name, h1, h1 - h0 + 1
        self.hist = it[(it.week >= h0) & (it.week <= h1)]
        self.views = d["views"][(d["views"].week >= h0) & (d["views"].week <= h1)]
        self.sched_hist = sc[(sc.week >= h0) & (sc.week <= h1)]
        self.sched_tgt = sc[(sc.week >= t0) & (sc.week <= t1)]
        self.n_tgt_weeks = t1 - t0 + 1
        self.cands = np.sort(self.sched_tgt.tv_show_id.unique())
        tgt = it[(it.week >= t0) & (it.week <= t1)]
        self.tgt = tgt
        self.truth = make_truth(tgt.rename(columns={"uid": "user_id"})) if len(tgt) else {}
        test = set(d["users"].uid[d["users"].test])
        self.users_all = np.array(sorted(self.truth)) if self.truth else np.array(sorted(test))
        self.users_test = np.array([u for u in self.users_all if u in test])
        self.pop = popularity(self.hist, self.cands)

    def evaluate(self, preds):
        return mapk(self.truth, preds, self.users_all), mapk(self.truth, preds, self.users_test)


def popularity(hist, cands, last_weeks=None):
    """Shows ranked by distinct users with >= 1 airing watched >= 80 %, restricted to the candidate pool."""
    h = hist[hist.frac >= 0.8]
    if last_weeks:
        h = h[h.week > h.week.max() - last_weeks]
    n = h.groupby("tv_show_id").uid.nunique()
    n = n[n.index.isin(cands)].sort_values(ascending=False, kind="stable")
    rest = [c for c in cands if c not in set(n.index)]
    return list(n.index) + rest


def fill(lst, pop, k=5):
    out = list(dict.fromkeys(lst))[:k]
    for p in pop:
        if len(out) >= k:
            break
        if p not in out:
            out.append(p)
    return out


def topk(scores, cands, users, pop, k=5):
    """scores: (len(users), len(cands)) array; ties and zero rows fall back to popularity order."""
    prank = np.empty(len(cands))
    pos = {c: i for i, c in enumerate(cands)}
    order = [pos[p] for p in pop if p in pos]
    prank[order] = np.arange(len(order))
    preds = {}
    for b in range(0, len(users), 2000):
        s = scores[b:b + 2000]
        idx = np.lexsort((np.broadcast_to(prank, s.shape), -s), axis=1)[:, :k]
        for u, row in zip(users[b:b + 2000], idx):
            preds[u] = [int(cands[j]) for j in row]
    return preds


def log_attempt(method, config, fold, a, b, secs, note=""):
    f = RES / "attempts.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "method", "config", "fold", "map5_all", "map5_test", "fit_predict_s", "note"])
        w.writerow([time.strftime("%H:%M:%S"), method, config, fold, f"{a:.5f}", f"{b:.5f}", f"{secs:.1f}", note])
    print(f"{method:10s} {config:45s} fold {fold}: MAP@5 all {a:.4f}  test {b:.4f}  ({secs:.1f}s)", flush=True)


def log_timing(method, config, fold, secs, n_users):
    f = RES / "timings.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["method", "config", "fold", "fit_predict_s", "n_users_scored"])
        w.writerow([method, config, fold, f"{secs:.1f}", n_users])


def write_submission(preds, name, pop):
    """preds: uid -> list of show ids. Users without a prediction get popularity. Runs the checker."""
    d = load()
    tpl = pd.read_csv(DATA / "submission.csv", dtype=str)
    m = dict(zip(d["users"].user_id, d["users"].uid))
    rows = [" ".join(str(x) for x in fill(preds.get(m[u], []), pop)) for u in tpl.user_id]
    out = pd.DataFrame({"user_id": tpl.user_id, "tv_show_id": rows})
    SUB.mkdir(exist_ok=True)
    path = SUB / f"{name}.csv"
    out.to_csv(path, index=False)
    r = subprocess.run([sys.executable, str(ROOT / "tools/check_submission.py"), str(path), str(DATA / "submission.csv"),
                        str(sorted(DATA.glob("export_arh_31-42*.csv"))[0])], capture_output=True, text=True)
    print(r.stdout.strip())
    return r.returncode == 0
