# Designated final submission (chosen by fold-A validation only)

**Designated final: `submissions/hybrid_chan+rel.csv`** (chosen 2026-11-01 15:04 MSK, same rule as before: best fold-A MAP@5 on
test users in results/attempts.csv). Claude Code has seen no Kaggle score; whether Maksim uploaded anything between
14:56 and 15:04 does not enter this choice.

Model: candidate union (user's history shows airing in weeks 31–42, top-30 PPR, top-30 ALS, top-30 rules, top-20 popularity,
top-50 channel affinity) re-ranked by LightGBM `lambdarank` (1 seed, 34 features), trained on fold-A + fold-B candidate
tables; features from history weeks 11–30; candidates = labelled shows of weeks 31–42.

| method | fold A all users | fold A test users |
|---|---|---|
| **hybrid + channel candidates + relative features (lambdarank)** | 0.3862 | **0.4258** |
| hybrid + channel candidates (lambdarank) | 0.3853 | 0.4258 (0.42579 vs 0.42580) |
| hybrid + channel + relative, fold-B rows added to CV training | 0.3873 | 0.4255 |
| hybrid v1 (4 candidate sources, lambdarank) — previous choice, `submissions/hybrid.csv` | 0.3813 | 0.4214 |
| hybrid v1 binary, 3 seeds | 0.3815 | 0.4195 |
| PPR alpha=0.7, weight=c80, restart=user (best single method) | 0.3460 | 0.3871 |
| repeat frequency (reference) | 0.3201 | 0.3631 |
| popularity (all history weeks) | 0.2465 | 0.2688 |

The first two rows are tied on test users (difference 1e-5); the relative features add +0.0009 on all users.

History: at 14:46 `submissions/hybrid.csv` (v1, 0.4214) was designated; at 15:04 it was replaced by the v2 file above after
the two improvement ideas were validated on fold A.

Upload order suggested (max 4): `popularity.csv` (calibration), `ppr.csv` (best single method), `hybrid_chan+rel.csv`
(designated final), `hybrid.csv` (v1, reserve).
