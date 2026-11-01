<!-- Drafted by Claude Code from results/; TODO-MAKSIM = Maksim writes in his own words. Numbers: results/*.csv|txt|md. -->

# Repeat-Aware Hybrid Recommendation of TV Programmes on sweet.tv: Personalized PageRank, ALS and Association Rules under a Schedule-Constrained MAP@5 Protocol

<div class="titleblock">
Student: Maksim Petrenko &nbsp;&nbsp; Team: Individual &nbsp;&nbsp; Email: madsm1006@gmail.com &nbsp;&nbsp; Date: 2026-11-01<br>
Assignment: Midterm Quiz — TV Program Recommendation (sweet.tv Kaggle competition)<br>
Code: https://github.com/TODO-MAKSIM/TODO-repo &nbsp;&nbsp; Kaggle user: TODO-MAKSIM
</div>

**Abstract** — We recommend five TV programmes per user for ISO weeks 31–42 of 2020 from sweet.tv channel-viewing logs of weeks 11–30 and the programme schedule, scored by MAP@5. Views are joined to labelled airings by channel and time overlap, and every method is validated on the same temporal split (history weeks 11–20, target weeks 21–30) with the competition's own label rule; we compare popularity, Personalized PageRank (PPR), implicit ALS and association rules, and combine them in a LightGBM re-ranker. A rank-averaged ensemble of four LightGBM re-rankers reaches MAP@5 0.4319 on the 1,238 test users of the validation fold, against 0.3871 for the best single method (PPR) and 0.2688 for popularity; its Kaggle private score is TODO-MAKSIM. Because the label counts repeatedly watched airings, methods that keep a user's own programmes on top dominate pure collaborative filtering.

*Index Terms* — recommender systems, implicit feedback, personalized PageRank, matrix factorization, association rules, MAP@5

## 1. Introduction
**Problem.** For each of 1,260 sweet.tv users we must output a ranked Top-5 of programme IDs (`tv_show_id`) for weeks 31–42. Inputs are a channel-level viewing log (who watched which channel from when to when, weeks 11–30) and the broadcast schedule (which programme aired on which channel when, weeks 11–42). The ground truth, per the competition Data tab [7], is each user's five programmes with the most airings watched for ≥ 80 % of their duration in the target window; the metric is MAP@5.
**Motivation.** A Top-5 "your programmes this week" row is the cheapest personalisation an IPTV service can put on its home screen; if it shows the shows a user actually returns to, it shortens the path to playback, which is what retention depends on.
**Example.** User `5024631139882254261` watched mostly cartoons in weeks 11–20 ('Вспыш и чудо-машинки' 143 airings ≥ 80 %, 'Свинка Пеппа' 140, 'Маша и Медведь' 66); the final hybrid's fold-A Top-5 is 'Свинка Пеппа' ✓, 'Вспыш…' ✓, 'Губка Боб', 'Элвин и бурундуки', 'Гуппи и пузырики' ✓ (AP@5 0.520, `results/examples.md`).
**Contributions.** (i) A single leakage-free protocol (join → temporal split → competition label rule → MAP@5) applied identically to five methods and 63 logged runs (`results/attempts.csv`); (ii) evidence that on this label, repeat-preserving graph scores beat matrix factorisation by a wide margin, and an ensemble of re-rankers with schedule-aware (channel-affinity) candidates adds +0.045 MAP@5 on top.

## 2. Related Work
- [1] Hu, Koren, Volinsky: implicit-feedback ALS with confidence weights c = 1 + α·r — our ALS confidence (seconds or ≥ 80 % counts).
- [2] Agrawal, Srikant: Apriori support/confidence and downward closure — our pairwise rules over "watched ≥ 80 %" baskets.
- [3] Haveliwala: PageRank with a personalised teleport vector — our PPR restart at the user (or uniform over the user's shows).
- [4] Covington et al.: two-stage candidate generation → ranking — the structure of our hybrid.
- [5] Ke et al.: LightGBM gradient-boosted trees — our re-ranker (binary and lambdarank objectives).
- [6] Frederickson: the `implicit` library — our ALS implementation.
- [7] Gurbych: the competition, data and label rule.

*Alternatives considered.* BPR-MF (pairwise ranking loss) — not used, ALS already shows that pure CF misses the repeat signal; item-kNN — PPR with restart at the user covers the same 2-hop neighbourhood; content-based filtering on genres/titles — dropped for time (2-hour session), category and duration enter the re-ranker as features instead.

## 3. Method
**Data and matching rule.** View log (`user_id, vsetv_id, start_time, stop_time, duraton`) × schedule (`channel_id, tv_show_id, start_time, duration, title, category, genres, …`). A view matches an airing if `vsetv_id = channel_id` and the intervals overlap; overlap seconds are summed per (user, airing) and clipped to the airing duration (24,006 of 3,014,129 pairs were clipped); screen time `frac = overlap / duration`; the week is the ISO week of the airing start. **Missing programme IDs:** rows with `tv_show_id = 0` (unlabelled airtime, 89 % of schedule rows, 83–84 % of airtime) are not items and are dropped before the join; 94.1 % of watched seconds land on some airing, 53.1 % on labelled airtime (`results/join_checks.txt`). `user_id` is a 64-bit hash, read as a string (never float). A timezone check (share of view starts within 60 s of an airing start for clock shifts −3…+3 h) peaks at 0 h (3.63 % vs ≤ 3.53 %), so no offset is applied; the signal is weak.

| | rows | users | channels / shows | note |
|---|---|---|---|---|
| view log weeks 11–30 | 5,821,987 | 4,914 | 202 channels | 3,014,129 (user, labelled airing) pairs |
| schedule file 11-20 (labelled / total) | 115,613 / 1,074,235 | | 400 ch. / 5,847 shows | labelled airtime 16.0 % |
| schedule file 21-30 (labelled / total) | 121,505 / 1,074,394 | | 403 ch. / 6,363 shows | labelled airtime 16.9 % |
| schedule file 31-42 (labelled / total) | 144,774 / 1,295,378 | | 404 ch. / 6,636 shows | candidate pool weeks 31–42: 6,642 shows |
| test users (submission.csv) | 1,260 | | | all in the view log; 1,258 with labelled history |

*Table 1. Data accounting (`results/data_stats.txt`, `results/final_window_stats.txt`). Files overlap at week borders, so all splits use the airing's ISO week.*

**Validation protocol (identical for all methods).** Fold A: history = joined rows of weeks 11–20 → truth = competition rule on weeks 21–30 (count airings with frac ≥ 0.8 per show, top-5 by count, ties by smaller id; `tools/evaluate.py`); candidate pool = labelled shows airing in weeks 21–30 (6,376); MAP@5 with denominator min(5, |truth|) as in Kaggle's ml_metrics. We report (a) all 4,643 users with non-empty truth and (b) the 1,238 test users among them. Users with no history and padding of short lists use popularity (history window, restricted to candidates). Fold B (11–24 → 25–30) checks robustness. Viewing data of the target weeks is used only to build the truth.

**Pipeline.** `prep.py` join → `cache/interactions.parquet` → `protocol.py` split / truth / candidates → `methods.py` scores (users × candidates) → Top-5 with popularity tie-break → `features.py` candidate union + features → `ranker.py` LightGBM → `run.py final` → `submissions/*.csv` → `tools/check_submission.py`.

**Methods.** *Popularity* (reference): distinct users with ≥ 1 airing watched ≥ 80 %. *Repeat frequency* (reference): the user's own shows by count of ≥ 80 % airings, ties by seconds. *PPR*: bipartite user–show graph, edge = log(1+seconds) or count of ≥ 80 % airings, row-normalised transitions, restart probability 1−α. Because the walk alternates user → show → user, the show part of the PPR vector is (1−α)·q·(I − α²T)⁻¹ with T = P(show→user)·P(user→show); we solve this exactly (one 7,146 × 7,146 inverse per configuration) instead of iterating, and checked it against 200 power iterations: max abs difference 7.0e-10 (`results/ppr_check.txt`). *ALS* [1], [6]: confidence 1 + a·log(1+sec/60) or 1 + a·count80, 15 iterations, factors {32, 64}, λ {0.01, 0.1}. *Association rules* [2]: baskets = shows with ≥ 1 airing watched ≥ 80 %; frequent shows (support ≥ 5 or 20 users), pairwise rules a→b with confidence ≥ 0.05 or 0.2; a candidate is scored by the max or sum of confidences (or lift) of rules fired by the basket, optionally with the trivial self-rule a→a = 1 so that own shows stay eligible. *Hybrid* [4], [5]: candidates = the user's history shows that air in the target window ∪ top-30 PPR ∪ top-30 ALS ∪ top-30 rules ∪ top-20 popularity ∪ top-50 by channel affinity Σ_c share(u, c)·airings(show on c in the target window) (187.2 per user, recall of truth 0.8428; 0.7909 without the channel source — `results/candidate_recall_*.txt`); features: per-week rates of ≥ 80 % airings and seconds, counts in the last 4 / 2 weeks, in-top-5-of-last-2-weeks flag, distinct days, weeks since last watch, share of the show's airings watched, airings of the show per target week, expected = share × airings, show popularity (per week, last 4 weeks), category, median duration, user's share of time on the show's main channel, hour-of-day and weekday profile overlap, user activity, PPR/ALS/rules/channel-affinity/repeat score and rank (rank 9999 for a zero score), popularity rank, share of the user's ≥ 80 % airings, rank of `expected` within the user, new-show flag, and recency features (≥ 80 % counts decayed with half-life 1 and 3 weeks, last-week count, seconds in the last 4 weeks, the user's ≥ 80 % airings in the last 4 weeks) — 39 features. Four objectives, each a LightGBM model, rank-averaged within each user in the final model: binary (label = show in the fold-A truth), lambdarank on the same label, lambdarank with graded labels (truth items 1–5 by log2 of their ≥ 80 % airings), and Poisson regression on the count of ≥ 80 %-watched airings in the target window — the quantity the label rule ranks by. 3-fold GroupKFold by user so no user is scored by a model that saw it.

**Tools & libraries** (`results/versions.txt`): Python 3.11.5, numpy 1.26.4, pandas 2.1.1, pyarrow 15.0.1, scipy 1.11.3, scikit-learn 1.2.2, lightgbm 4.7.0, implicit 0.7.3, threadpoolctl 3.6.0; macOS 14.5, Apple M1 Pro, 16 GB RAM.

**Key design decisions.** (1) PPR edges count ≥ 80 %-watched airings, because the label counts exactly those (log-seconds edges: best 0.3055 vs 0.3871, test users). (2) History counts are per-week rates (plus fixed-length last-4/2-week counts), because fold A has 10 history weeks and the final run 20; the final ranker is trained on fold A and fold B (14 weeks) candidate tables. (3) Candidates are restricted to shows that air in the target window — a show that is not broadcast cannot be watched.

**Configuration.** Seeds 42 (ALS, LightGBM; 3-seed variant 42–44); LightGBM 400 trees, learning rate 0.05, 63 leaves, min 50 samples per leaf, 0.8 row/column subsampling. Fit+predict times in `results/timings.csv` (Table 2).

## 4. Experiments
**Setup.** Fold A as in §3 (4,643 users with truth, 1,238 of them test users, 6,376 candidates); fold B for the best configuration of each method; hardware as above.

| Method (best config of its grid) | MAP@5 fold A (all, n=4,643) | MAP@5 fold A (test, n=1,238) | fold B (test) | fit + predict | Kaggle public / private |
|---|---|---|---|---|---|
| Popularity (all history weeks) | 0.2465 | 0.2688 | 0.2742 | 0.6 s | TODO-MAKSIM |
| Popularity (last 4 weeks) | 0.2747 | 0.3022 | – | 0.4 s | – |
| Repeat frequency (reference) | 0.3201 | 0.3631 | 0.3413 | 1.8 s | – |
| PPR α=0.7, edge=count80, restart=user | 0.3460 | 0.3871 | 0.3576 | 9.9 s | TODO-MAKSIM |
| ALS 64 factors, λ=0.1, conf=1+count80 | 0.1908 | 0.1923 | 0.2045 | 2.9 s | – |
| Rules sup≥20, conf≥0.2, max, self-rule | 0.2739 | 0.2973 | 0.2904 | 1.8 s | – |
| Hybrid v1: LightGBM lambdarank, 4 candidate sources | 0.3813 | 0.4214 | – | 57.8 s (CV) / 54.0 s (final) | TODO-MAKSIM or – |
| + channel-affinity candidates | 0.3853 | 0.4258 | – | 63.0 s (CV) | – |
| + within-user relative features | 0.3862 | 0.4258 | – | 58.8 s (CV) / 61.6 s (final) | TODO-MAKSIM or – |
| + recency features (lambdarank alone) | 0.3851 | 0.4232 | – | 61.0 s (CV) | – |
| 4-objective rank-average ensemble (final) | 0.3903 | **0.4319** | – | 4 CV runs 49–63 s each / 136.6 s (final) | TODO-MAKSIM |

*Table 2. Same fold, users, candidates and metric for every row (`results/attempts.csv`, 63 runs incl. 12 PPR, 6 ALS, 10 rules, 27 hybrid runs; `results/timings.csv`; CV times include the 22 s candidate build). Not chosen: hybrid v1 binary 0.4172 (1 seed) / 0.4195 (3 seeds); fold-B rows added to CV training 0.4255; Poisson / graded / binary alone 0.4213 / 0.4248 / 0.4219; bigger LightGBM (1,000 trees, 127 leaves) 0.4166; channel top-100 candidates (recall 0.8662) ensemble 0.4282.*

**Comparison vs baseline.** Every personalised method except ALS beats popularity; PPR adds +0.118 MAP@5 (test users) over popularity and +0.024 over the user's own repeat frequency, and the re-ranker ensemble adds a further +0.045 (single lambdarank v1 +0.034; the channel-affinity candidate source lifts the candidate ceiling from 0.7909 to 0.8428 and MAP@5 by +0.0044, the within-user relative features add +0.0009 on all users and nothing on test users; rank-averaging the four objectives adds +0.006 on test users and +0.004–0.005 on all users over the best single model, and the ensemble gain repeats on the other two feature sets tried: 0.4292 vs 0.4258 without recency features, 0.4282 vs 0.4246 with channel top-100 candidates). Fold B preserves the single-method ranking on both user sets (PPR > repeat > rules > popularity > ALS; all users 0.3236 > 0.3078 > 0.2667 > 0.2479 > 0.1954). In a binary re-ranker on the final feature set, channel-affinity rank and score carry 34 % of the gain and PPR rank and score 30 % (`results/feature_importance_chan+rec+rel.csv`).

**Two users (Table 3, full tables in `results/examples.md`).** TODO-MAKSIM: one sentence each on why these two users. *Heavy viewer* `5024631139882254261` (666 airings ≥ 80 % in weeks 11–20, 18 distinct items across the six Top-5 lists): history is children's cartoons plus 'ТСН. Неделя'; truth = 'Свинка Пеппа', 'Вспыш…', 'Моланг', two more cartoons. Repeat 0.550, PPR 0.400 and the hybrid 0.520 return the user's cartoons (the hybrid ranks 'Свинка Пеппа' and 'Вспыш…' first, then 'Губка Боб' and 'Элвин' (misses), and finds 'Гуппи и пузырики' (✓), which the single methods miss); ALS (0.000) recommends 'Кухня', 'Варьяты', 'Секретные материалы' — what similar households watch, not what this one repeats; rules (0.000) put popular news first because all the user's own shows tie at the self-rule confidence 1 and popularity breaks the tie. *Near-cold* `11191705376024801445` (2 airings ≥ 80 %, both 'Голос страны', which does not air in weeks 21–30): truth = 'Кухня' + four feature films; only repeat finds 'Кухня' (rank 5, AP 0.040); all other methods score 0 — a single partially watched history cannot predict four films.

**Verification.** (i) `tools/evaluate.py` self-test (lecture example AP values) passes (`results/evaluate_selftest.txt`); (ii) PPR closed form = power iteration to 7.0e-10; (iii) the join was checked on one view record (`results/hand_check.md`: view 20:59:15–22:04:50 on channel 1495 overlaps three cartoon airings by 1,200 s, 1,500 s and 1,190 s); (iv) one AP@5 laid out step by step in `results/hand_check.md`; (v) every submission passes `tools/check_submission.py` (1,260 rows, 5 distinct non-zero ids from the weeks 31–42 schedule). TODO-MAKSIM: state which of (iii)/(iv) you recomputed by hand and the result; Kaggle score of `popularity.csv` vs its validation 0.2688.

**Kaggle.** Figure 1: Submissions page with the designated final submission (`submissions/hybrid_chan+rec+rel_ens4.csv`, chosen by fold-A MAP@5 — `results/final_choice.md`; private score TODO-MAKSIM).
![Kaggle submissions page](figures/kaggle_submissions.png)

## 5. Discussion
TODO-MAKSIM — write in your own words from `results/discussion_notes.md`: strategy for sweet.tv (validation + cost + examples); how the methods differ and what it means for users; one failure case (log excerpt → root cause → fix → verification); what worked / surprised; limitations; next improvement.

## 6. AI Usage Disclosure
- Tools: Claude Code 2.1.316 (model Claude Opus 5.5), plus Claude (chat) for preparing the task kit (CLAUDE.md, PROMPT.md, tools/, references.md).
- How AI was used: code generation (all of `src/`), debugging (Kaggle CLI 403, the quadratic `set()` hang in `run.py final`), data checks and report drafting (all numbers copied from `results/`); search was not used — references come from the pre-vetted `references.md`.
- What I personally verified: TODO-MAKSIM.
- What I trusted without verification: TODO-MAKSIM.
- Session log: `petrenko_maksim_midterm_session.json` (Claude Code transcript, unmodified), duration TODO-MAKSIM.

## References
<ol class="refs">
<li>Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, Pisa, Italy, 2008, pp. 263–272, doi: 10.1109/ICDM.2008.22. [Online]. Available: https://dblp1.uni-trier.de/rec/bibtex/conf/icdm/HuKV08. [Accessed: 2026-10-29].</li>
<li>R. Agrawal and R. Srikant, "Fast algorithms for mining association rules in large databases," in *Proc. 20th Int. Conf. Very Large Data Bases (VLDB)*, Santiago de Chile, 1994, pp. 487–499. [Online]. Available: https://vldb.org/dblp/db/conf/vldb/vldb94-487.html. [Accessed: 2026-10-29].</li>
<li>T. H. Haveliwala, "Topic-sensitive PageRank," in *Proc. 11th Int. World Wide Web Conf. (WWW 2002)*, Honolulu, HI, USA, 2002, doi: 10.1145/511446.511513. [Online]. Available: https://archives.iw3c2.org/www2002/CDROM/refereed/127.html. [Accessed: 2026-10-29].</li>
<li>P. Covington, J. Adams, and E. Sargin, "Deep neural networks for YouTube recommendations," in *Proc. 10th ACM Conf. Recommender Systems (RecSys)*, Boston, MA, USA, 2016, doi: 10.1145/2959100.2959190. [Online]. Available: https://research.google/pubs/pub45530. [Accessed: 2026-10-29].</li>
<li>G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, Long Beach, CA, USA, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf. [Accessed: 2026-10-29].</li>
<li>B. Frederickson, "implicit: Fast Python collaborative filtering for implicit datasets," GitHub repository. [Online]. Available: https://github.com/benfred/implicit. [Accessed: 2026-10-29].</li>
<li>O. Gurbych, "sweet.tv – TV program recommender," Kaggle community competition, 2020. [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender. [Accessed: TODO-MAKSIM date].</li>
</ol>
