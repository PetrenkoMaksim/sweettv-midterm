<!--
TEMPLATE for report/report.md  (Claude Code: copy this file to report/report.md and fill it).
Structure = course homework guidelines §5 (IEEE-aligned) + the midterm's extra requirements.
HARD LIMIT: 5 pages excluding References. Keep tables compact (8.5pt). Every number must be copied
from a log / results CSV that exists in the repo (name the file next to the number the first time).
Markers:  TODO = Claude Code fills.   TODO-MAKSIM = Maksim rewrites in his own words before building the PDF
(scripts/make_pdf.sh refuses to build while any TODO remains).
-->

# [Concise paper-style title, e.g. "Repeat-Aware Hybrid Recommendation of TV Programmes on sweet.tv: PPR, ALS and Association Rules under a Schedule-Constrained MAP@5 Protocol"]

<div class="titleblock">
Student: Maksim Petrenko &nbsp;&nbsp; Team: Individual &nbsp;&nbsp; Email: madsm1006@gmail.com &nbsp;&nbsp; Date: TODO (YYYY-MM-DD)<br>
Assignment: Midterm Quiz — TV Program Recommendation (sweet.tv Kaggle competition)<br>
Code: https://github.com/TODO-MAKSIM/TODO-repo &nbsp;&nbsp; Kaggle user: TODO-MAKSIM
</div>

**Abstract** — (50–150 words, 4 sentences) (1) Problem: Top-5 TV programmes per user for weeks 31–42 from channel-viewing logs of weeks 11–30 and the programme schedule, scored by MAP@5. (2) Method: schedule join, temporal validation with the competition's own label rule, popularity baseline, PPR / ALS / association rules, hybrid re-ranker. (3) Key result with ONE specific number (validation MAP@5 of the final method vs the popularity baseline; Kaggle private score). (4) One takeaway. TODO

*Index Terms* — recommender systems, implicit feedback, personalized PageRank, matrix factorization, association rules, MAP@5

## 1. Introduction
- Problem statement (2–4 sentences): what is predicted, for whom (1,260 users), from what (view log + schedule), metric. TODO
- Motivation (2–3 sentences): why Top-5 "most-watched" programmes matter for an IPTV service (home screen, retention). TODO
- Concrete example (one input → output pair: a real user id, 3–4 programmes from their history, the Top-5 the final method returns). TODO
- Contributions (1–2 bullets, no over-claim). TODO

## 2. Related Work
3–7 entries from references.md, IEEE numbered, each with a one-clause statement of what was taken from it. TODO
Alternatives considered (1–3, one line each with rationale): e.g. BPR-MF, item-kNN, two-tower; why not used here. TODO

## 3. Method
**Data and matching rule.** View log (user_id, vsetv_id, start_time, stop_time) × schedule (channel_id, tv_show_id, start_time, duration, title, category, genres). Rule: same channel (`vsetv_id = channel_id`) and overlapping time interval; overlap seconds summed per (user, airing) and clipped to the airing duration; screen time = overlap / duration. **Missing programme IDs:** rows with `tv_show_id = 0` are unlabelled airtime — dropped before the join (they are not items), while the channel-level time still feeds channel-affinity features. Report the counts actually loaded and dropped (Table 1). TODO

| | views / rows | users | channels / shows | note |
|---|---|---|---|---|
| view log weeks 11–30 | TODO | TODO | TODO | |
| schedule 11–30 (labelled / total) | TODO | | TODO | share of labelled airtime TODO % |
| schedule 31–42 (labelled / total) | TODO | | TODO | candidate pool |
| test users (submission.csv) | 1,260 | | | TODO with history / TODO cold |

**Validation protocol (identical for all methods).** Fold A: history weeks 11–20 → targets weeks 21–30; truth = competition rule (screen time ≥ 0.8, count airings per show, top-5 by count); candidate pool = labelled shows airing in the target weeks; MAP@5 with denominator min(5, |R_u|) (ml_metrics). Users without history → popularity fallback. No viewing data from the target weeks is used for training. TODO: fold B if used.

**Pipeline** (step list or small diagram): join → interactions → split → candidates → methods → (hybrid) → submission. TODO

**Methods.** One short paragraph each (what, key hyper-parameters, cost): popularity baseline (reference only); Personalized PageRank (graph, α, iterations); ALS implicit MF (factors, regularization, confidence); association rules (support/confidence/lift thresholds, scoring); CBF if implemented; hybrid (candidate generation + LightGBM re-ranker: feature list, objective, folds). TODO

**Tools & libraries** (with versions from `results/versions.txt`): TODO

**Key design decisions** (2–3 non-obvious choices + rationale, e.g. counting airings instead of seconds because the label rule counts airings; normalising history counts per week because the final history is twice as long as fold A's; restricting to shows that air in the target window). TODO

**Configuration**: seeds, hardware, run times (`results/timings.csv`). TODO

## 4. Experiments
**Setup.** Dataset sizes, protocol (one sentence pointing to §3), hardware. TODO

**Results (Table 2).** All methods, same fold, same users; include the baseline and every variant that was tried (no cherry-picking: `results/attempts.csv` has the full list).

| Method | MAP@5 fold A (all users with truth) | MAP@5 fold A (test users) | fit + predict time | Kaggle public / private |
|---|---|---|---|---|
| Popularity | TODO | TODO | TODO | TODO / TODO |
| Personal repeat frequency (reference) | TODO | TODO | TODO | – |
| Personalized PageRank | TODO | TODO | TODO | – |
| ALS (implicit MF) | TODO | TODO | TODO | – |
| Association rules | TODO | TODO | TODO | – |
| Hybrid re-ranker (final) | TODO | TODO | TODO | TODO / TODO |

**Comparison vs baseline** (2–3 sentences, numbers). TODO

**Two users (Table 3).** Why these two (one heavy viewer on whom the methods disagree; one near-cold user) — TODO-MAKSIM justifies the choice in one sentence each. Same users for every method. For each user: history summary (top watched shows with categories), then each method's Top-5 with programme ID, title, category; mark hits against the fold-A truth; compare with the viewing history and programme attributes. TODO

**Verification.** Cite specific checks: hand-computed AP@5 for one user (show the arithmetic), the join checked by hand on one view record, the submission file checked by `tools/check_submission.py`, the Kaggle score of the popularity baseline compared with the validation number. TODO (facts) — TODO-MAKSIM confirms he did at least one by hand.

**Kaggle.** Figure 1: screenshot of the Submissions page with the designated final submission (file `submissions/TODO.csv`, private score TODO).
![Kaggle submissions page](figures/kaggle_submissions.png)

## 5. Discussion
*(Most heavily weighted section. Drafted from the logs by Claude Code; TODO-MAKSIM rewrites every paragraph in his own words before the PDF is built.)*
- **Strategy selected for sweet.tv and why**: validation performance, computational cost (per-user cost of PPR vs ALS vs rules vs hybrid, numbers from timings), concrete recommendation examples from Table 3. TODO
- **How and why the methods differ and what it means for the user experience** (repeat-heavy lists vs discovery; popularity vs personalisation; cold users). TODO
- **Failure case** (one real one: an error message / log excerpt, e.g. a wrong join, a float-cast of user_id, a leak) → **root cause** → **fix + verification**. TODO
- **What worked** (2–3 bullets) / **what surprised or didn't work** (2–3 bullets). TODO
- **Limitations and failure cases supported by results** (label rule only reconstructed; tie-break unknown; 12-week horizon vs 10-week fold; cold users; schedule drift between training and target weeks; public vs private gap). TODO
- **Next improvement** with more time (one concrete item). TODO

## 6. AI Usage Disclosure
- Tools: Claude Code (model + version from `claude --version`), plus Claude (chat) for preparing the task kit. TODO
- How AI was used (search / code generation / debugging / docs / viz) — list by category with 1 line each. TODO
- What I personally verified (for each critical output: how) — TODO-MAKSIM (truthfully; e.g. hand AP@5 for user X, re-ran `python src/run.py validate` and got the same table, read [k] §… ).
- What I trusted without verification — TODO-MAKSIM (honest list).
- Session log: `petrenko_maksim_midterm_session.json` (Claude Code transcript, unmodified), duration TODO.

## References
<ol class="refs">
<li>TODO — copy entries verbatim from references.md (only the ones cited), IEEE format, with [Accessed: YYYY-MM-DD].</li>
</ol>
