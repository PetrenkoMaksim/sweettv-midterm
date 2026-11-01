# sweet.tv TV-programme recommender — LLM4Rec midterm (Maksim Petrenko)

Top-5 TV programmes per user for ISO weeks 31–42 of 2020 from the channel-viewing log of weeks 11–30 and the programme
schedule (Kaggle community competition `sweettv-tv-program-recommender`, metric MAP@5).

## Reproduce
```bash
pip install -r requirements.txt          # macOS: brew install libomp first (LightGBM)
bash scripts/download_data.sh            # Kaggle CLI; competition rules must be accepted on the website
python src/prep.py                       # join views x labelled schedule -> cache/*.parquet       (~30 s)
python src/run.py validate               # fold A: baselines, PPR / ALS / rules grids, fold B check, hybrid CV (~6 min)
python src/run.py final                  # history 11-30 -> submissions/*.csv (+ checker)             (~1 min; add 'singles' to rewrite the single-method files)
python src/run.py examples               # results/examples.md, results/hand_check.md (~30 s)
python src/run.py baselines              # quick popularity.csv + repeat.csv only (~6 s)
```
Every validation run appends rows to `results/attempts.csv` (all configurations tried) and `results/timings.csv`.

## Layout
| path | content |
|---|---|
| `src/prep.py` | join (same channel, interval overlap, summed per (user, airing), clipped to duration) + data/join checks |
| `src/protocol.py` | folds (A: 11–20 → 21–30, B: 11–24 → 25–30, final: 11–30 → 31–42), truth rule, candidate pool, MAP@5, popularity fill, submission writer |
| `src/methods.py` | popularity, personal repeat, Personalized PageRank, ALS (`implicit`), pairwise association rules |
| `src/features.py`, `src/ranker.py` | hybrid: candidate union + LightGBM re-ranker |
| `src/examples.py` | two-user tables and the hand-check sheet |
| `tools/evaluate.py`, `tools/check_submission.py` | MAP@5 (Kaggle ml_metrics convention) + truth rule; submission format check |
| `results/` | every number used in the report |
| `submissions/` | submission files; the designated final one is named in `results/final_choice.md` |

## Designated final submission
`submissions/hybrid_chan+rec+rel_ens4.csv`: rank-average of four LightGBM re-rankers (binary, lambdarank, graded lambdarank,
Poisson) over channel-affinity-extended candidates; fold-A MAP@5 0.4319 on test users. Chosen by fold-A validation — see
`results/final_choice.md`. Re-create it: `python src/run.py hybrid --opts chan,rel,rec --runs lambdarank:base,poisson:base,lambdarank_graded:base,binary:base`
(~4 min, writes `results/hybrid_choice.json`), then `python src/run.py final` (~2.5 min). Score saved ensembles:
`python src/run.py ens <opts> obj:size,obj:size`.

## Environment
See `results/versions.txt`.
