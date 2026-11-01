#!/usr/bin/env bash
# report/report.md -> deliverables/petrenko_maksim_midterm_report.pdf
# Tries in order: pandoc+xelatex, pandoc+weasyprint, markdown+Chrome headless. Refuses to build with placeholders left.
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=report/report.md
OUT=deliverables/petrenko_maksim_midterm_report.pdf
mkdir -p deliverables
[ -f report/figures/kaggle_submissions.png ] || { echo "MISSING report/figures/kaggle_submissions.png (Kaggle Submissions page screenshot)"; exit 1; }
grep -q "github.com/" "$SRC" || { echo "no GitHub link in $SRC"; exit 1; }
if grep -n -i "TODO\|FILL\|XXX" "$SRC"; then echo "placeholders left in $SRC"; exit 1; fi
rm -f "$OUT"

if command -v pandoc >/dev/null && command -v xelatex >/dev/null; then
  (cd report && pandoc report.md -o "../$OUT" --pdf-engine=xelatex -V geometry:margin=2cm -V fontsize=10pt) 2>/tmp/make_pdf.err \
    || { echo "pandoc+xelatex failed ($(grep -m1 Error /tmp/make_pdf.err)); trying the next engine"; rm -f "$OUT"; }
fi
if [ ! -f "$OUT" ] && command -v pandoc >/dev/null && python3 -c "import weasyprint" 2>/dev/null; then
  (cd report && pandoc report.md -o "../$OUT" --pdf-engine=weasyprint --css=report.css --metadata pagetitle=report) 2>/tmp/make_pdf.err \
    || { echo "pandoc+weasyprint failed; trying Chrome"; rm -f "$OUT"; }
fi
if [ ! -f "$OUT" ]; then
  python3 -c "import markdown" 2>/dev/null || python3 -m pip install -q markdown
  python3 - <<'EOF'
import markdown, pathlib
md = pathlib.Path("report/report.md").read_text()
css = pathlib.Path("report/report.css").read_text()
html = markdown.markdown(md, extensions=["tables", "fenced_code", "footnotes"])
pathlib.Path("report/report.html").write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>")
EOF
  CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  [ -x "$CHROME" ] || CHROME=$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)
  [ -n "$CHROME" ] || { echo "no working pandoc engine and no Chrome: brew install pandoc && brew install --cask mactex-no-gui (or pip install weasyprint)"; exit 1; }
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/$OUT" "file://$PWD/report/report.html" 2>/dev/null
fi
[ -f "$OUT" ] || { echo "PDF was not produced"; exit 1; }
python3 - <<'EOF'
try:
    import pypdf
    n = len(pypdf.PdfReader("deliverables/petrenko_maksim_midterm_report.pdf").pages)
    print(f"PDF pages: {n} (limit: 5 excluding references)")
except ImportError:
    print("PDF written (pip install pypdf to auto-count pages)")
EOF
echo "wrote $OUT"
