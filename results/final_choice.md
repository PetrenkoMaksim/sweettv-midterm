# Designated final submission (chosen by validation, before any Kaggle score existed)

Written 2026-11-01 ~14:46 MSK; no file had been uploaded to Kaggle at this time.

**Designated final: `submissions/hybrid.csv`** — candidate union (user's own history shows airing in weeks 31–42, top-30 PPR,
top-30 ALS, top-30 rules, top-20 popularity) re-ranked by LightGBM `lambdarank` (1 seed), trained on the fold-A and fold-B
candidate tables, features from history weeks 11–30, candidates = labelled shows of weeks 31–42.

Rule: best fold-A MAP@5 on test users (results/attempts.csv):

| method (best config) | fold A all users | fold A test users |
|---|---|---|
| hybrid, lgbm lambdarank, 1 seed | 0.3813 | **0.4214** |
| hybrid, lgbm binary, 3 seeds | 0.3815 | 0.4195 |
| hybrid, lgbm binary, 1 seed | 0.3794 | 0.4172 |
| PPR alpha=0.7, weight=c80, restart=user (best single method) | 0.3460 | 0.3871 |
| repeat frequency (reference) | 0.3201 | 0.3631 |
| popularity (all history weeks) | 0.2465 | 0.2688 |

Note: on "all users" the binary 3-seed run is 0.0002 higher, on test users lambdarank is 0.0019 higher — the two are
practically tied; the test-user column decides by the rule fixed in PROMPT.md.

Upload order suggested (max 4): `popularity.csv` (calibration), `ppr.csv` (best single method), `hybrid.csv` (designated
final), one reserve (`repeat.csv`).
