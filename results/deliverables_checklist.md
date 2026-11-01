# Deliverables vs. the exam slides (task/midterm_quiz.pdf = the shared Google Slides, identical file)

| Requirement (slides) | Where it is satisfied | Who |
|---|---|---|
| ≥ 2 methods of CBF / association rules / MF / PPR | PPR, ALS (MF), association rules — `src/methods.py`, Table 2 | done |
| Join logs ↔ schedule; matching rule; missing programme IDs | report §3 "Data and matching rule", `src/prep.py`, `results/join_checks.txt` | done |
| Same temporal split, candidate pool, protocol | report §3 "Validation protocol", `src/protocol.py` | done |
| Popularity baseline (reference, not counted) | Table 2 | done |
| Final may be hybrid; each method evaluated alone first | Table 2 rows per method, then hybrid rows | done |
| Program metadata used; no future viewing for training | §3 "Programme metadata used"; target-window views only build the truth | done |
| Strategy for sweet.tv and why | §5 — facts in `results/discussion_notes.md` | **Maksim writes** |
| Two users, justified, same users for all methods | Table 3 (+ `results/examples.md`); justification sentence | **Maksim: 1 sentence each** |
| Each method's Top-5 with IDs, titles, categories vs history & attributes | Table 3 + paragraph after it | done |
| How/why methods differ; UX implications | §4 "What users would see" + §5 | facts done, **Maksim writes §5** |
| Justify final strategy: validation, cost, examples | Table 2 (MAP + time), Table 3, §5 | **Maksim writes §5** |
| Limitations and failure cases supported by results | `results/discussion_notes.md` → §5 | **Maksim writes §5** |
| PDF report | `bash scripts/make_pdf.sh` → `deliverables/petrenko_maksim_midterm_report.pdf` (≤ 5 pages excl. references) | **Maksim** |
| Original, unmodified AI-session export | `bash scripts/export_session.sh` → `deliverables/petrenko_maksim_midterm_session.json` | **Maksim** |
| PDF: GitHub link | title block `TODO-MAKSIM` | **Maksim** |
| PDF: screenshot of the highest returned private score of the designated final submission | `report/figures/kaggle_submissions.png` (Figure 1) | **Maksim** |
| Repo: implementation code | `src/`, `tools/` | done (push) |
| Repo: reproduction instructions (optional) | `README.md` | done |
| Repo: the final submission file used for the reported score | `submissions/hybrid_chan+rec+rel_ens4.csv` (committed) | done (push) |
| Kaggle submission description = full name (Russian) & Kaggle name | `Петренко Максим (underqualified)` | **Maksim** |
| ≤ 5 Kaggle submissions ("so be wise") | plan: 4 uploads (`results/final_choice.md`) | **Maksim** |
| Deadline FKN: 1 Nov 19:00 | — | **Maksim** |

Grading reminder: 50 pts = rank of the returned (private) score of the designated final submission among exam participants;
50 pts = homework rubric c1/c2 (Gate 0: invalid reference, fabricated number, non-running code, claimed-but-not-done
verification → 0).
