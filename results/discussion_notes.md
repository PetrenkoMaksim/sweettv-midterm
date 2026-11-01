# Facts for §5 Discussion (Maksim writes the prose; every number below is from the named file)

## Strategy choice (attempts.csv, timings.csv, final_choice.md)
- Fold A MAP@5, test users: hybrid lambdarank 0.4214; best single method PPR (α=0.7, edge = count of ≥80 % airings,
  restart = user) 0.3871; repeat frequency 0.3631; rules best 0.2973; ALS best 0.1923; popularity 0.2688 (last 4 weeks: 0.3022).
- Fold B (history 11–24 → 25–30) keeps the order: PPR 0.3576 > repeat 0.3413 > rules 0.2904 > popularity 0.2742 > ALS 0.2045
  (test users). No method collapses.
- Cost (fit+predict, M1 Pro, 4,643 users): repeat 1.8 s; rules ~2 s; ALS 2.9–6.0 s; PPR ~10 s (dominated by the dense
  inverse of the 7,146 × 7,146 show-show matrix); hybrid 3-fold CV 57.8 s incl. 18 s candidate+feature build; final hybrid
  (fit on 1,600,782 rows, predict 400,212 rows for 1,258 users) 54 s.
- Hybrid gain over PPR: +0.034 MAP@5 (test users), +0.035 (all users). Over popularity: +0.153.
- Feature importance (feature_importance.csv, gain share): ppr_rank 0.356, ppr_score 0.135, air_tgt_rate 0.082
  (airings of the show in the target window), expected 0.060, category 0.038, chan_share 0.033.

## How the methods differ → UX (examples.md, submission_overlap.txt)
- Label = most frequently ≥80 %-watched shows → rewards *repeat* viewing. Methods that keep the user's own shows on top
  (repeat, PPR with count edges) win; ALS (pure CF reconstruction, own shows not privileged) recommends "what similar
  users watch" → heavy user 5024631139882254261 (kids' cartoons) gets 'Кухня', 'Варьяты', 'Секретные материалы' from ALS, AP@5 0.
- Rules best config uses max-confidence with the self-rule a→a = 1: all own frequent shows tie at 1, tie broken by
  popularity → the heavy kids' viewer gets news ('ТСН. Неделя', 'Факты') from rules, AP@5 0. Sum aggregation is lower on fold A
  (0.2815 vs 0.2973).
- Popularity = same 5 news/entertainment programmes for everybody (AP@5 0 on both example users).
- Mean overlap of the final Top-5 lists over 1,260 users: hybrid–PPR 3.80 / 5, hybrid–repeat 3.40, PPR–repeat 3.78
  → the hybrid mostly re-orders PPR/repeat lists and swaps ~1 item.
- Near-cold user 11191705376024801445: their only ≥80 % show ('Голос страны', 2 airings) does not air in weeks 21–30 → no
  method can repeat it; truth = 'Кухня' + 4 films (Х/ф) the user had not watched ≥80 %. Repeat gets 'Кухня' at rank 5 (AP 0.04).

## Real failure cases from this session (decisions.md, final_log.txt)
1. Final hybrid run hung: `python src/run.py final` produced no output for > 10 min; `sample <pid>` first showed LightGBM
   frames, a second sample showed the Python eval loop in `pandas IndexEngine.__contains__`. Root cause:
   `ub = np.array([u for u in FB.users_all if u in set(FB.hist.uid)])` — the set of ~2 M uids is rebuilt for each of ~4.6 k
   users (quadratic). Fix: `np.intersect1d(FB.users_all, FB.hist.uid.unique())`. Verification: the rerun finished in
   58 s wall (final_log.txt) and `hybrid.csv` passed tools/check_submission.py. (A first guess — merged lambdarank groups
   because fold-A and fold-B rows share uids — was real but not the hang; fold-B uids are now offset by 10⁶.)
2. PPR "restart on own shows" with edge-weighted restart gave identical numbers to restart-at-user (synthetic smoke test).
   Root cause: q = P(user→show)[u] is exactly the first step of the user restart. Fix: uniform restart over own shows.
   On real data the uniform variant is worse (0.3426 vs 0.3809 at α=0.5, c80).
3. Data access: Kaggle CLI 1.5.12 `403 - Forbidden`; CLI 2.2.5 "accept its rules first" → rules accepted by Maksim at 14:27.

## What worked / surprised
- Count-of-≥80 %-airings edges ≫ log-seconds edges for PPR (0.3871 vs 0.3055 best, test users): the edge weight that mirrors
  the label wins.
- ALS is the weakest method (≤ 0.1923, below popularity); larger confidence scale a=5 collapses it to 0.0321; 64 factors are
  worse than 32 with log-seconds confidence (0.0857 vs 0.1219).
- Lift as the rule score collapses (0.0688): lift promotes rare shows.
- "Popularity of the last 4 weeks" beats all-history popularity (0.3022 vs 0.2688) — viewing volume drops strongly in weeks
  19–27 (data_stats.txt: 407 k rows in week 11 vs 105 k in week 27) → recency matters.

## Limitations (backed by results)
- Candidate recall 0.7909 (candidate_recall.txt) → ~21 % of truth items can never be ranked by the hybrid.
- Fold A has 10 history weeks, the final run 20; count features are per-week rates, last-4/last-2-week counts are absolute.
  Final model trained on fold A (10 weeks) + fold B (14 weeks) rows; target window 10 (A) / 6 (B) vs 12 weeks (final).
- Timezone check (join_checks.txt) peaks at 0 h shift but weakly (3.63 % of view starts within 60 s of an airing start vs
  3.21–3.53 % for ±1..3 h).
- Truth tie-break of the host unknown (we use smaller id). Only 53 % of watched seconds land on labelled airtime.
- No Kaggle score yet at the time of writing; validation-vs-Kaggle gap unknown until Maksim uploads popularity.csv.

## Next improvement
- Raise candidate recall (0.79): e.g. add shows of the user's top channels airing in the target window and top-N by
  `expected` (share watched × airings in target window) as a candidate source.

## Improvement round (15:00–15:04, results/attempts.csv rows with opts=…)
- Diagnosis (scratch script, numbers printed in the session): 4,498 of 21,513 truth items (20.9 %) were outside the v1
  candidates; 27.4 % of those were shows with no history edges at all (new in the target window) → unreachable for PPR/ALS/rules.
- Idea 1, channel-affinity candidates (Σ_c user's watch-time share on c × airings of the show on c in the target window,
  top-50): recall 0.7909 → 0.8428, MAP@5 test 0.4214 → 0.4258 (+0.0044), all 0.3813 → 0.3853. chan_rank becomes the top
  feature (gain share 0.226).
- Idea 2, within-user relative features (repeat score/rank, share of the user's ≥80 % airings, rank of `expected` within the
  user, new-show flag, rank 9999 for zero scores): all +0.0009, test +0.0000 → essentially no gain.
- Extra check: fold-B rows added to CV training (as the final model does): all 0.3873, test 0.4255 → neutral.
- Final v2 file differs from v1 by 0.65 of 5 items per user on average (submission_overlap.txt: 4.349 shared).

## Improvement round 2 (15:20–15:34; attempts.csv rows 44–64)
- Objectives alone (feature set chan+rec+rel, test users): lambdarank 0.4232, graded lambdarank 0.4248, binary 0.4219,
  Poisson on the target count 0.4213 → none beats lambdarank chan+rel (0.4258).
- Rank-average of the four: 0.4319 test / 0.3903 all → +0.006 / +0.004 over the best single model. The gain repeats on the
  other feature sets (chan+rel 0.4292, chan100+rec+rel 0.4282); lambdarank + Poisson alone already 0.4299 → the Poisson
  model (fits the count, not membership) is the most complementary member.
- Recency features: no gain alone (0.4232 vs 0.4258), but the ensemble on that table is the best.
- Bigger LightGBM (1,000 trees, lr 0.03, 127 leaves): worse (lambdarank 0.4166, binary 0.4188) and 2.5× slower.
- Channel top-100 candidates: recall 0.8428 → 0.8662 but MAP@5 not higher (ensemble 0.4282) — the extra candidates are
  low-probability items the ranker rarely lifts into the top 5.
- Reproducibility: the chan+rel lambdarank run repeated at 15:24 gave exactly 0.38619 / 0.42580 as at 15:00.
- Final ensemble: 136.6 s fit+predict (4 models on 1,769,870 rows), vs 61.6 s for the single lambdarank.
- Noise caveat: 1,238 test users → differences of a few 0.001 are not meaningful; only the ensemble gain (+0.006, repeated on
  three feature sets) is consistent.

## UX / business facts (results/ux_stats.csv, results/truth_repeat_share.txt; computed on the submitted files, 1,260 users)
- Share of Top-5 slots that are shows the user already watched ≥ 80 % in weeks 11–30: hybrid ens4 0.892, PPR 0.938,
  ALS 0.944, rules 0.988, repeat 0.988, popularity 0.726.
- Share of slots the user never touched (pure discovery): hybrid ens4 0.028, PPR 0.029, ALS 0.021, repeat 0.002, popularity 0.103.
- Catalogue coverage (distinct shows over all 1,260 lists): popularity 5, hybrid ens4 112, rules 150, ALS 157, PPR 190, repeat 438.
- Share of slots in the global top-20 popularity list: rules 0.944, PPR 0.795, hybrid ens4 0.761, ALS 0.658, repeat 0.632.
- Ground truth (fold A): only 54.4 % of a user's future top-5 had been watched ≥ 80 % before, 71.5 % touched at all, 28.5 %
  never touched → the models (≈ 89–99 % repeats) are more repeat-heavy than the users; the missing ~28 % of "new" favourites
  is where the remaining MAP is (and where a discovery slot in the UI would help).
- Cost for a weekly batch: final hybrid 136.6 s for 1,258 users incl. fitting (timings.csv); PPR 9.9 s for 4,643 users;
  popularity < 1 s — all are offline batch jobs, no per-request latency concern at this scale.

## Discussion length budget (measured by rendering the report with a placeholder Discussion and screenshot)
- ≤ ~700 words in §5 keeps all content before "References" within 5 pages; at 900 words it reaches the limit.
