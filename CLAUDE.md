# Project rules — LLM4Rec midterm (sweet.tv Kaggle recommender)

Student: Maksim Petrenko (HSE, LLM4Rec, instructor Seungmin Jin). Task: `task/midterm_quiz.pdf`.
Grading rules: `task/homework_guidelines.pdf` (Gate 0 = automatic fail). Read both once at the start.

## What is graded (and how to not fail)
- 50 pts: rank among exam participants by the **returned private score** of the designated final Kaggle submission (MAP@5).
- 50 pts: homework rubric c1 (problem + valid references) and c2 (working solution + accurate reasoning + personal verification).
- **Gate 0 (score 0 for everything)**: hallucinated / wrong citation, dead URL, fabricated or cherry-picked number, code that does not run,
  claimed verification that was not done, API/function that does not exist, generic 3-bullet discussion.
  Therefore: every number in the report is copied from a file in `results/` produced by code in `src/`; every reference comes from
  `references.md`; failed attempts are listed, not hidden; nothing is described as "verified by Maksim" unless it is in the hand-check sheet he can do.

## Data facts (verified on other students' public repos — re-check against the real files on first load)
- Files (names differ from the exam PDF!): `data/dataset11-30.csv` (views), `data/export_arh_11-20-final.csv`, `data/export_arh_21-30-final.csv`,
  `data/export_arh_31-42-final.csv` (schedule), `data/submission.csv` (template). Use `glob` on `dataset11-30*` / `export_arh_*` to be safe. ~890 MB.
- Views: `user_id, vsetv_id, start_time, stop_time, duraton` (sic). ~5.8M rows, ISO weeks 11–30 of 2020 (week 11 starts Mon 2020-03-09 — confirm).
  `start_time` is ISO-like; **user_id is a 64-bit hash: read as `uint64` or `str`, never let it become float** (precision loss corrupts the submission).
- Schedule: `channel_id, tv_show_id, start_time ("%d.%m.%Y %H:%M:%S"), duration (s), tv_show_title, tv_show_category, tv_show_genre_1..3,
  director, actors, year_of_production, ...` (print `df.columns` and confirm). ~1M rows per file; only ~11 % of rows / airtime are labelled.
  `tv_show_id = 0` = unlabelled airtime → not an item: drop before the join. `vsetv_id` (views) == `channel_id` (schedule).
- Template: 1,260 users, column `tv_show_id` = 5 space-separated ids (placeholder "0 0 0 0 0").
  **Checked 2026-11-01 (results/data_stats.txt): all 1,260 appear in the view log** (not ~1,150). Views: 5,821,987 rows, 4,914 users,
  202 channels, `start_time` = "YYYY-MM-DD HH:MM:SS", weeks 11–30 confirmed. Schedule has an extra `channel_title` column; files overlap
  at week borders (11-20 file spans ISO weeks 10..21) → always split by the airing's ISO week, never by file. Labelled rows ≈ 11 %,
  labelled airtime ≈ 16–17 %.
- Ground-truth rule published on the competition Data tab (quoted by participants; Maksim must confirm the wording on the Data tab himself):
  *"all entries with tv_show_id = 0 were removed from TV program schedule files; for each user and each TV show user screen time (time user
  watched the show divided by total duration of the show) was calculated. Only TV shows with high screen time (>= 0.8) were taken into account;
  for each user high-screen time TV shows frequency was calculated; for each user strictly 5 most frequent TV shows were selected DESC"*.
  → the label is "the 5 programmes with the most ≥80 %-watched airings in weeks 31–42". Validate with exactly this rule (`tools/evaluate.py: make_truth`).
- Reference points reported publicly by other participants: popularity ≈ 0.28 private; personal repeat frequency / PPR ≈ 0.39; LightGBM hybrid ≈ 0.43–0.45.
  Target: hybrid ≥ 0.44 on fold A. Validation with the rule above tracked the private score within ~0.01 for them.

## Protocol (same for every method)
- Join: same channel + interval overlap; overlap seconds summed per (user, airing), clipped to duration; `frac = ov / dur`; week = ISO week of the airing start.
- Fold A (primary): history = joined rows of weeks 11–20 → truth = rule above on weeks 21–30; candidates = labelled shows airing in weeks 21–30 (schedule only).
  Report MAP@5 on (a) all users with non-empty truth, (b) the test users among them. Fold B (11–24 → 25–30) only as a robustness check.
- Final: history weeks 11–30, candidates = labelled shows airing in weeks 31–42 (`export_arh_31-42-final.csv`). Never use any viewing data from the target window
  (there is none for 31–42; in fold A, the 21–30 views are only used to build the truth).
- MAP@5 as in `tools/evaluate.py` (Kaggle ml_metrics). Cold users → popularity Top-5 (computed on the history window, restricted to candidates).
- Distribution shift: fold A has 10 weeks of history, the final run 20. Express count features per week of history (rates) or cap them; say so in the report.

## Engineering rules
- Python 3.11+, pandas/numpy/scipy/scikit-learn; LightGBM for the ranker (`brew install libomp` if import fails); `implicit` for ALS if it installs
  cleanly, otherwise a 40-line NumPy ALS (Hu et al. 2008) — the item/user counts are small enough for dense factors.
- Code style: simple and short; moderate file sizes; very short comments; no over-engineering, no CLI frameworks, no excessive function arguments.
  Layout: `src/prep.py` (join → `cache/*.parquet`), `src/protocol.py` (split, truth, candidates, metric — may import `tools/evaluate.py`),
  `src/methods.py` (popularity, repeat, ppr, als, rules, cbf), `src/features.py` + `src/ranker.py` (hybrid), `src/run.py` (validate / final / examples),
  `src/examples.py` (two-user tables). Cache the join (parquet) — the join takes a minute, everything else must iterate in seconds.
- Seeds fixed (42). Record wall-clock of fit+predict per method into `results/timings.csv`. Save every validation attempt (method, config, MAP@5 a/b, time)
  as a row of `results/attempts.csv` — this table is the no-cherry-picking evidence.
- Every submission file is written to `submissions/<name>.csv` and passed through `tools/check_submission.py` before anyone uploads it.
- `data/` and `cache/` are git-ignored; `submissions/`, `results/`, `src/`, `report/`, `tools/` are committed. Commit at every milestone.
- Do not read, copy or adapt other students' repositories for this exam (not even for "inspiration"): the code must be original to this session.

## Honesty rules for the report
- Claude Code drafts facts; Maksim writes §5 Discussion and the "personally verified / trusted" parts of §6 in his own words.
  Leave `TODO-MAKSIM` markers where his text must go and prepare `results/hand_check.md` with the raw inputs he needs to check one AP@5 by hand
  and one join by hand (the view record, the matching airings, the overlap arithmetic).
- No invented library versions: write `results/versions.txt` from `pip freeze` / `python -V` and copy from it.
- Only references from `references.md`, only the entries actually used, verbatim.

## Time budget and checkpoints
Total budget: see PROMPT.md. Produce a Kaggle-ready popularity + repeat-frequency submission in the first hour, so there is always something to upload.
Do not ask questions that block the run; make the reasonable choice, write it in `results/decisions.md`, and continue.
Kaggle uploads are done by Maksim (or by Claude Code via the CLI only if PROMPT.md says so); at most 4 uploads in total are planned, so be wise.
