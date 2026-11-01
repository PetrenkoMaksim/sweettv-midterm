# Midterm kit — how to run it

Deadline (FKN groups): **Sat 1 Nov 2026, 19:00 MSK**. Submit the PDF report + the unmodified session export; the GitHub repo and the Kaggle submission are linked/screenshotted inside the PDF.

## 0. Before starting Claude Code (10 min)
1. Kaggle: log in, open https://www.kaggle.com/competitions/sweettv-tv-program-recommender, click **Late Submission** once (accept rules). Create an API token (Settings → API → Create New Token) → `~/.kaggle/kaggle.json`, `chmod 600 ~/.kaggle/kaggle.json`. Read the **Data** tab once and check that the ground-truth sentence quoted in `CLAUDE.md` is really there (it is your citation [12]).
2. Mac prerequisites: `brew install libomp` (LightGBM), `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. For the PDF: `brew install pandoc` plus either `brew install --cask mactex-no-gui` or `pip install weasyprint`; otherwise Google Chrome is used.
3. Unzip the kit into e.g. `~/Desktop/midterm_claude_code_kit`, open `PROMPT.md`, fill the FILL-IN block (Kaggle username, whether Claude Code may upload itself).
4. `cd` into the folder, `git init`, run `claude`, paste the content of `PROMPT.md`. Use **one** session for the whole exam (use `/compact` if the context fills; avoid `/clear` or starting a second session — the transcript is the audit trail). If you must continue in a new session, submit both transcripts.

## 1. While it runs
Check `results/decisions.md` and `results/attempts.csv` now and then. Override the two example users if you prefer others (`results/examples.md` lists alternatives).

## 2. Kaggle uploads (max 4; late-submission counters allow more, but keep it clean)
Order: `submissions/popularity.csv` (calibrates validation vs Kaggle), best single method, `submissions/hybrid.csv`, one reserve.
Description field = **"Петренко Максим (<kaggle username>)"** — the instructor matches you by this. Check the Submissions page shows your name correctly (see the "Error case" slide).
Designated final = the one chosen in `results/final_choice.md` by validation; the private score "returned" for it is what is ranked. The slides say the instructor accepts the best score among your uploads, so if another file scores higher you may designate that one — then say in the report that the choice used Kaggle feedback (honest disclosure, not a Gate 0). Then take the screenshot: Submissions page, your name visible, the final file's private score visible → save as `report/figures/kaggle_submissions.png`.

## 3. GitHub
`gh repo create sweettv-midterm --public --source=. --push` (or create the repo on github.com and push). Must contain `src/`, `tools/`, `submissions/` (incl. the final file), `results/`, `README.md`; NOT `data/` or `cache/`. Put the URL into `report/report.md` (title block) and in `README.md`.

## 4. Finish the report (your part, ~45 min)
Open `report/report.md`, search `TODO-MAKSIM`:
- justify the two example users (one sentence each);
- §5 Discussion in your own words — use `results/discussion_notes.md` for the facts; keep the failure-case → root cause → fix → verification chain;
- §6: what you verified yourself (do the hand check in `results/hand_check.md`: compute one AP@5 by hand and re-run `python src/run.py validate` once — then you can honestly list both), what you trusted;
- access date for the Kaggle reference, GitHub link, Kaggle user; delete every remaining marker.
Then `bash scripts/make_pdf.sh` → `deliverables/petrenko_maksim_midterm_report.pdf`. Check: ≤ 5 pages excluding references, screenshot visible, every reference URL opens in your browser (click them all — a dead link is Gate 0).

## 5. Session export and submission
`bash scripts/export_session.sh` → `deliverables/petrenko_maksim_midterm_session.json` (a byte-for-byte copy of the Claude Code transcript; do not edit it).
Submit `deliverables/petrenko_maksim_midterm_report.pdf` + `deliverables/petrenko_maksim_midterm_session.json` through the course channel.

## What is in this kit
| file | purpose |
|---|---|
| `PROMPT.md` | the message for Claude Code (staged plan, 4 h budget — change it in the FILL-IN block) |
| `CLAUDE.md` | standing rules Claude Code reads automatically: grading traps, data facts, protocol, code style |
| `task/midterm_quiz.pdf`, `task/homework_guidelines.pdf` | the exam sheet and the course rubric (Gate 0 definition) |
| `tools/evaluate.py` | MAP@5 (Kaggle ml_metrics convention, self-tested against the lecture example) + the competition's truth rule |
| `tools/check_submission.py` | format check before any upload |
| `references.md` | 12 references, links opened on 2026-10-29, IEEE format — you still click each one before submitting |
| `report/report_template.md`, `report/report.css` | report skeleton (course §5 structure + midterm items), print style |
| `scripts/download_data.sh`, `scripts/export_session.sh`, `scripts/make_pdf.sh` | data download via Kaggle CLI, transcript copy, PDF build |
| `requirements.txt`, `.gitignore` | environment and what not to commit |
