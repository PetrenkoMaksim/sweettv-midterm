# Designated final submission (chosen by fold-A validation only)

**Designated final: `submissions/hybrid_chan+rec+rel_ens4.csv`** (chosen 2026-11-01 15:34 MSK by the same rule as before:
best fold-A MAP@5 on test users in results/attempts.csv; it is also the best on all users). Claude Code has seen no Kaggle
score; whatever Maksim uploaded in between does not enter this choice.

Model: candidate union (user's history shows airing in weeks 31–42, top-30 PPR, top-30 ALS, top-30 rules, top-20 popularity,
top-50 channel affinity), 39 features (channel affinity, within-user relative, recency), four LightGBM models — binary,
lambdarank, lambdarank with graded labels, Poisson on the target count — rank-averaged within each user. Trained on fold-A +
fold-B candidate tables; features from history weeks 11–30; candidates = labelled shows of weeks 31–42.
(The file was written as `hybrid_chan+rec+rel_lambdarank-base_poisson-base_lambdarank_graded-base_binary-base.csv` and renamed;
`src/run.py final` now writes the short name directly.)

| hybrid variant (fold A) | all users | test users |
|---|---|---|
| **4-objective rank-average ensemble, opts chan+rec+rel** | **0.3903** | **0.4319** |
| 4-objective ensemble, opts chan+rel (no recency) | 0.3886 | 0.4292 |
| 4-objective ensemble, opts chan100+rec+rel | 0.3893 | 0.4282 |
| lambdarank, opts chan+rel — previous choice `hybrid_chan+rel.csv` | 0.3862 | 0.4258 |
| lambdarank, opts none — first choice `hybrid.csv` | 0.3813 | 0.4214 |
| PPR (best single method) | 0.3460 | 0.3871 |
| popularity | 0.2465 | 0.2688 |

History of the designation: 14:46 `hybrid.csv` (0.4214) → 15:04 `hybrid_chan+rel.csv` (0.4258) → 15:34 this file (0.4319).

Upload order suggested (max 4): `popularity.csv` (calibration), `ppr.csv` (best single method),
`hybrid_chan+rec+rel_ens4.csv` (designated final), `hybrid_chan+rel.csv` (reserve, single lambdarank).
