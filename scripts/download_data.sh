#!/usr/bin/env bash
# Downloads the competition files into ./data (needs the Kaggle CLI + ~/.kaggle/kaggle.json,
# and the competition rules accepted on the website first).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if ls data/dataset11-30*.csv >/dev/null 2>&1; then echo "data already present"; exit 0; fi
command -v kaggle >/dev/null || python3 -m pip install kaggle
kaggle competitions download -c sweettv-tv-program-recommender -p data
cd data && unzip -o -q ./*.zip && rm -f ./*.zip && ls -la
