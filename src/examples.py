"""Two fold-A test users: history summary + every method's Top-5 vs truth -> results/examples.md, results/hand_check.md."""
import json

import numpy as np
import pandas as pd

import methods
from protocol import RES, ROOT, Fold, apk, load, topk

USERS = None  # set to (heavy_uid, cold_uid) to override the automatic choice


def method_preds(F, M, users):
    best = json.loads((RES / "best_configs.json").read_text())
    out = {"popularity": {u: F.pop[:5] for u in users},
           "repeat": topk(methods.repeat(F, M, users), F.cands, users, F.pop),
           "ppr": topk(methods.ppr(F, M, users, **best["ppr"]), F.cands, users, F.pop),
           "als": topk(methods.als(F, M, users, **best["als"]), F.cands, users, F.pop),
           "rules": topk(methods.rules(F, M, users, **best["rules"])[0], F.cands, users, F.pop)}
    tag = json.loads((RES / "hybrid_choice.json").read_text()).get("opts", "none")
    oof = pd.read_csv(ROOT / f"cache/hybrid_A_oof_{tag}.csv", index_col=0).iloc[:, 0]
    out["hybrid"] = {u: [int(x) for x in oof[u].split()] for u in users}
    return out


def main():
    F = Fold("A")
    M = methods.Mat(F)
    meta = load()["sched"].groupby("tv_show_id")[["tv_show_title", "tv_show_category", "tv_show_genre_1"]].first()
    users = F.users_test
    n80 = np.asarray(M.c80[users].sum(1)).ravel()
    P = method_preds(F, M, users)
    dis = np.array([len(set(sum((P[m][u] for m in P), []))) for u in users])
    top = n80 >= np.quantile(n80, 0.9)
    heavy = list(users[top][np.argsort(-dis[top], kind="stable")][:3])
    cold = [u for u, n in zip(users, n80) if 1 <= n <= 3][:3]
    lines = ["# Example users (fold A: history weeks 11–20, truth weeks 21–30)", "",
             "Candidates offered (first = chosen; the others are alternatives Maksim may pick by setting `USERS` in src/examples.py):", ""]
    for lab, cs in (("heavy viewer (top-10 % by >=80 % airings, max distinct items across methods' Top-5)", heavy),
                    ("near-cold (1–3 airings watched >= 80 % in history)", cold)):
        for u in cs:
            i = list(users).index(u)
            lines.append(f"- {lab}: uid {u} (user_id {uid2id(u)}), >=80 % airings {int(n80[i])}, "
                         f"distinct items across methods {dis[i]}, truth size {len(F.truth[u])}")
    chosen = USERS or (heavy[0], cold[0])
    hand = None
    for u in chosen:
        lines += ["", f"## user_id {uid2id(u)} (uid {u})", "", "History (weeks 11–20), top shows by >=80 %-watched airings:", "",
                  "| tv_show_id | title | category | genre 1 | airings >=80 % | watched min |", "|---|---|---|---|---|---|"]
        h = F.hist[F.hist.uid == u]
        g = h.groupby("tv_show_id").agg(n80=("frac", lambda x: int((x >= 0.8).sum())), mins=("ov", "sum"))
        for s, r in g.sort_values(["n80", "mins"], ascending=False).head(8).iterrows():
            lines.append(f"| {s} | {meta.tv_show_title.get(s)} | {meta.tv_show_category.get(s)} | "
                         f"{meta.tv_show_genre_1.get(s)} | {r.n80} | {r.mins / 60:.0f} |")
        tr = F.truth[u]
        lines += ["", f"Truth (weeks 21–30, competition rule): {tr}", "",
                  "| method | AP@5 | Top-5 (id · title · category; ✓ = in truth) |", "|---|---|---|"]
        for m, pr in P.items():
            items = " <br> ".join(f"{'✓' if s in tr else '·'} {s} · {meta.tv_show_title.get(s)} · {meta.tv_show_category.get(s)}"
                                  for s in pr[u])
            lines.append(f"| {m} | {apk(tr, pr[u]):.3f} | {items} |")
        if hand is None:
            hand = (u, P["hybrid"][u], tr, F)
    (RES / "examples.md").write_text("\n".join(lines) + "\n")
    write_table(F, P, chosen, meta)
    write_hand_check(*hand)
    ux_stats(F, M)
    print("\n".join(lines))


def short(title, n=24):
    t = str(title)
    for pre in ("М/с ", "Т/с ", "Х/ф ", "Д/с ", "Д/ф "):
        t = t.removeprefix(pre)
    t = t.replace("'", "").replace('"', "").split(",")[0].rstrip(".")
    return t if len(t) <= n else t[:n - 1] + "…"


def write_table(F, P, chosen, meta):
    """Compact Table 3 for the report: per user history, truth and every method's Top-5 (id, short title, category)."""
    cell = lambda s, tr: f"{'✓ ' if s in tr else ''}{s} {short(meta.tv_show_title.get(s))} [{meta.tv_show_category.get(s)}]"
    L = []
    for k, u in enumerate(chosen):
        h = F.hist[F.hist.uid == u]
        g = h.groupby("tv_show_id").agg(n80=("frac", lambda x: int((x >= 0.8).sum())), mins=("ov", "sum"))
        g = g.sort_values(["n80", "mins"], ascending=False).head(4)
        tr = F.truth[u]
        L += [f"**User {'A (heavy)' if k == 0 else 'B (near-cold)'}, `{uid2id(u)}`.** History weeks 11–20 (airings ≥ 80 %): " +
              "; ".join(f"{short(meta.tv_show_title.get(s))} [{meta.tv_show_category.get(s)}] {r.n80}" for s, r in g.iterrows()) +
              ". Truth weeks 21–30: " + "; ".join(f"{s} {short(meta.tv_show_title.get(s))} [{meta.tv_show_category.get(s)}]" for s in tr) + ".",
              "", "| method | AP@5 | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|---|"]
        for m, pr in P.items():
            L.append(f"| {m} | {apk(tr, pr[u]):.3f} | " + " | ".join(cell(s, tr) for s in pr[u]) + " |")
        L.append("")
    (RES / "examples_table.md").write_text("\n".join(L))


def ux_stats(F, M):
    """What the final lists look like for users: repeats vs new shows, catalogue coverage; and how repeat-heavy the truth is."""
    from protocol import SUB, DATA
    d = load()
    it = d["inter"][d["inter"].week <= 30]
    seen80 = set(zip(it.uid[it.frac >= 0.8], it.tv_show_id[it.frac >= 0.8]))
    seen = set(zip(it.uid, it.tv_show_id))
    m = dict(zip(d["users"].user_id, d["users"].uid))
    pop20 = set(Fold("final").pop[:20])
    rows = []
    for f in sorted(SUB.glob("*.csv")):
        s = pd.read_csv(f, dtype=str)
        pairs = [(m[u], int(x)) for u, l in zip(s.user_id, s.tv_show_id) for x in l.split()]
        rows.append({"file": f.name, "distinct_shows": len({p[1] for p in pairs}),
                     "share_watched80_before": np.mean([p in seen80 for p in pairs]),
                     "share_never_watched": np.mean([p not in seen for p in pairs]),
                     "share_in_pop_top20": np.mean([p[1] in pop20 for p in pairs])})
    pd.DataFrame(rows).round(3).to_csv(RES / "ux_stats.csv", index=False)
    h = F.hist
    s80 = set(zip(h.uid[h.frac >= 0.8], h.tv_show_id[h.frac >= 0.8]))
    s = set(zip(h.uid, h.tv_show_id))
    tp = [(u, x) for u, l in F.truth.items() for x in l]
    (RES / "truth_repeat_share.txt").write_text(
        f"fold A truth items (weeks 21-30, {len(tp)} user-show pairs): watched >=80 % in weeks 11-20 "
        f"{np.mean([p in s80 for p in tp]):.3f}; touched at all in weeks 11-20 {np.mean([p in s for p in tp]):.3f}; "
        f"never touched {np.mean([p not in s for p in tp]):.3f}\n")


def uid2id(u):
    us = load()["users"]
    return us.user_id[us.uid == u].iloc[0]


def write_hand_check(u, pred, tr, F):
    it = load()["inter"]
    t = it[(it.uid == u) & (it.week >= 21) & (it.week <= 30)]
    c = t[t.frac >= 0.8].groupby("tv_show_id").size().sort_values(ascending=False)
    L = ["# Hand-check sheet (raw inputs; Maksim computes the results himself)", "",
         f"## (i) AP@5 of the hybrid (fold A, out-of-fold) for user_id {uid2id(u)}", "",
         f"Airings watched >= 80 % per show in weeks 21–30 (the truth rule counts these): {c.head(8).to_dict()}", "",
         f"Truth (top-5 by count, tie-break by smaller id): {tr}", f"Prediction (rank 1..5): {pred}", "",
         "Rule (tools/evaluate.py apk): walk the list; at each hit at rank i add (hits so far)/i; divide by min(|truth|, 5).", ""]
    hits, s, steps = 0, 0.0, []
    for i, p in enumerate(pred, 1):
        if p in tr:
            hits += 1
            s += hits / i
            steps.append(f"rank {i}: {p} hit -> +{hits}/{i}")
        else:
            steps.append(f"rank {i}: {p} miss")
    L += [f"- {x}" for x in steps]
    L += ["", f"AP@5 = {s:.4f} / {min(len(tr), 5)} = {s / min(len(tr), 5):.4f} (code: {apk(tr, pred):.4f})", ""]
    L += [(RES / "hand_check_join.md").read_text(), "",
          "Arithmetic: overlap = min(view stop, airing end) − max(view start, airing start); frac = overlap / airing duration.", "",
          "## (iii) Re-run the fold-A table", "", "```", "python src/prep.py      # ~1–2 min, builds cache/",
          "python src/run.py validate   # appends a new block of rows to results/attempts.csv; compare with the existing rows", "```"]
    (RES / "hand_check.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
