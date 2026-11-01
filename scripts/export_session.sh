#!/usr/bin/env bash
# Copies the Claude Code session transcript of THIS project folder, unmodified, into deliverables/.
# Claude Code stores transcripts at ~/.claude/projects/<escaped-cwd>/<session-id>.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="petrenko_maksim_midterm"
ROOT="$HOME/.claude/projects"
PROJ=""
for cand in "$ROOT/$(pwd | sed 's#[^A-Za-z0-9]#-#g')" "$ROOT/$(pwd | sed 's#/#-#g')"; do
  [ -d "$cand" ] && PROJ="$cand" && break
done
if [ -z "$PROJ" ]; then
  echo "project transcript folder not found by name; using the most recently modified one:"
  PROJ=$(ls -1dt "$ROOT"/*/ | head -1)
fi
echo "transcript folder: $PROJ"
mkdir -p deliverables
ls -1S "$PROJ"/*.jsonl >/dev/null 2>&1 || { echo "no .jsonl transcripts in $PROJ"; exit 1; }
BIG=$(ls -1S "$PROJ"/*.jsonl | head -1)
cp "$BIG" "deliverables/${NAME}_session.json"
echo "copied $(basename "$BIG") -> deliverables/${NAME}_session.json ($(du -h "$BIG" | cut -f1))"
N=$(ls -1 "$PROJ"/*.jsonl | wc -l | tr -d ' ')
if [ "$N" -gt 1 ]; then
  echo "NOTE: $N transcripts exist here; the largest was copied. Others:"
  ls -1S "$PROJ"/*.jsonl | tail -n +2
  echo "If more than one session was used for the exam, submit all of them (zip), unmodified."
fi
