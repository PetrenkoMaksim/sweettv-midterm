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
