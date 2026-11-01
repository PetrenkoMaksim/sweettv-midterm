# Decisions log (Claude Code, running)

Times are MSK, 2026-11-01. Session start 13:22.

- 13:22 Budget: the user's message says **2 h** (PROMPT.md FILL-IN says 4 h) → the shorter one is used; stage deadlines scaled
  ×0.5 (S1 ≤ 0:30, S2 ≤ 1:15, S3 ≤ 1:45, S4 ≤ 2:00). CBF is out of scope (prompt: only if time remains after the hybrid).
- 13:22 Kaggle upload permission in PROMPT.md is left as "YES / NO" (not chosen) → treated as **NO**: Claude Code writes and checks
  files, Maksim uploads.
- 13:23 `scripts/download_data.sh` failed: the Homebrew Kaggle CLI 1.5.12 returned `403 - Forbidden`. Upgraded the CLI in the
  conda Python (`pip install -U kaggle` → 2.2.5); file listing then worked, but the download returned
  "You don't have access to this resource … you may need to accept its rules first". Accepting competition rules is a legal
  agreement on Maksim's account → not done by Claude Code; Maksim was asked to accept them. A background loop retries the
  download every 30 s.
- 13:24–13:33 While blocked: wrote `src/prep.py`, `protocol.py`, `methods.py`, `features.py`, `ranker.py`, `run.py`,
  `examples.py` against the schema in CLAUDE.md, and smoke-tested the full pipeline (prep → validate → final → examples)
  on a small **synthetic** dataset with the same columns (generated in the session scratchpad, never in `data/`; the
  `KIT_ROOT` env variable points the code at it). No synthetic number is used anywhere in the report.
- Join: views and labelled airings packed into one sorted key (channel·10¹⁰ + unix seconds); a cumulative max of airing
  ends gives the first airing that can still overlap a view, `searchsorted` on starts the last one → all overlapping pairs
  without a Python loop. Overlap summed per (user, airing), clipped to the airing duration.
- PPR: power iteration over ~all users + shows per batch was estimated at minutes per configuration. Because the graph is
  bipartite, the show part of the PPR vector has the closed form (1−α)·q·(I − α²T)⁻¹ with T = P(show→user)·P(user→show)
  (show × show, small). `methods.ppr` uses that; `methods.ppr_power` (plain power iteration) is kept to check it
  numerically → `results/ppr_check.txt`.
- PPR "restart on the user's own shows" weighted by the edge weights is *exactly proportional* to restart-at-the-user after
  one step (q = α·P(user→show)[u]) → identical rankings (seen on the synthetic run). The variant therefore uses a
  **uniform** restart over the user's shows.
- Association rules: Apriori with itemsets of size ≤ 2 (frequent single shows first — downward closure — then pairs among
  them). Variants: min support (users), min confidence, aggregation max/sum, with/without the trivial self-rule a→a
  (score 1 for the user's own shows, so repeats stay eligible).
- ALS: `implicit` 0.7.3 imports cleanly → used (`AlternatingLeastSquares`, the matrix values are the confidences
  c = 1 + a·log(1+sec/60) or 1 + a·count80). BLAS threads limited to 1 inside fit (implicit's own warning).
- 14:27 Maksim accepted the rules; the retry loop started the download. 65 min of the 2 h budget were lost to the block →
  grids trimmed before seeing any real result: ALS 6 configs (factors {32,64} × reg {0.01,0.1} with log-seconds confidence,
  + count-based confidence, + a=5), rules 10 configs, PPR full 12 (cheap closed form). Fold B only for the best config of
  each method. CBF dropped.
- 14:28 Real data: schema as in CLAUDE.md (+ `channel_title`); all 1,260 template users are in the view log (CLAUDE.md said
  ~1,150 → corrected). prep 28 s. Safety net `popularity.csv` + `repeat.csv` written and checked at 14:29.
- 14:29–14:35 `run.py validate` (fold A grids, fold B check, hybrid CV). Best: PPR α=0.7 count80 restart=user 0.3871 (test);
  ALS weak (≤ 0.1923); hybrid lambdarank 0.4214 → designated final (results/final_choice.md, written before any upload).
  Popularity of the last 4 weeks beats all-history popularity (0.3022 vs 0.2688) but the fill stays all-history for
  all methods (protocol fixed before results; changing it now would alter every row).
- 14:35–14:52 FAILURE: `run.py final` hung (no output > 10 min). First `sample` showed LightGBM frames → guessed merged
  lambdarank groups (fold-A and fold-B rows share uids) → offset fold-B uids by 10⁶ (correct but not the cause); rerun hung
  again, second `sample` showed the Python loop in pandas `IndexEngine.__contains__` → root cause
  `[u for u in FB.users_all if u in set(FB.hist.uid)]` rebuilds a ~2 M-element set per user. Fix: `np.intersect1d`.
  Rerun: 58 s wall, `hybrid.csv` OK by the checker.
- 14:53 examples: heavy viewer uid 3424 / near-cold uid 368 chosen automatically (alternatives listed in examples.md).
- Kaggle uploads: none done by Claude Code (permission not given in PROMPT.md).
- 14:57 Maksim asked to "try 2 highest-ROI ideas". Diagnosis: 20.9 % of truth items outside candidates, 27 % of them shows with
  no history edges → idea 1 = channel-affinity candidate source (top-50), idea 2 = within-user relative features.
  Hybrid artifacts are now tagged by option set (`cache/hybrid_A_<opts>.parquet`, `results/feature_importance_<opts>.csv`);
  `hybrid_choice.json` is only overwritten by a better fold-A test MAP@5.
- 15:00 idea 1: 0.4258 test (+0.0044); idea 2 on top: +0.0000 test / +0.0009 all. 15:03 fold-B rows in CV training: neutral
  (0.4255). 15:04 `run.py final` → `submissions/hybrid_chan+rel.csv` (checker OK) = new designated final.
- 15:05 Error caught while editing the report: I typed fold-B "all users" numbers (0.2479 vs 0.2408) that are not in
  attempts.csv and drew a wrong conclusion (rules < popularity); the file says rules 0.2667 > popularity 0.2479 → fixed.
  The same wrong "exception" was in my 14:56 chat summary.
- 15:06 Maksim: "Try more ideas … time budget 1 hour". Implemented as flags: objectives poisson (target = count of ≥80 %
  airings in the target window, stored as `cnt`, never a feature) and lambdarank_graded; LightGBM size "big"; recency
  features (opt rec); channel top-100 (opt chan100); rank-average ensemble; `run.py ens` scores subsets from saved OOF scores.
- 15:23 chan+rec+rel, 4 objectives: singles ≤ 0.4248, ensemble 0.4319 (best). 15:30 chan+rel with big models: big worse;
  ensembles 0.4278 (6 runs) / 0.4292 (4 base). 15:34 chan100+rec+rel: ensemble 0.4282. Stopped experimenting at 15:34.
- 15:36 `run.py final` → ensemble of 4 on chan+rec+rel, 136.6 s, checker OK → renamed to `hybrid_chan+rec+rel_ens4.csv`
  (naming code shortened afterwards); new designated final.
- 16:11 Maksim shared the requirements Google Slides; exported as PDF/text: byte-identical to task/midterm_quiz.pdf
  (same SHA-1). Audit → gaps fixed: (1) the per-method Top-5 tables for the two users (IDs, titles, categories) were only in
  results/, now Table 3 in the report (generated: results/examples_table.md); (2) UX facts added (results/ux_stats.csv,
  truth_repeat_share.txt); (3) the report states which programme metadata is used. Report trimmed and measured: with a
  700-word Discussion and the screenshot, content stays ≤ 5 pages. Reference italics fixed (HTML list, text unchanged).
  Checklist: results/deliverables_checklist.md.
