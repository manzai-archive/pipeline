#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — set HF_TOKEN before running diarization."
fi

echo
echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
echo "Run:           python -m pipeline ingest <url>"
