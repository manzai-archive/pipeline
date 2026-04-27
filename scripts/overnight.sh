#!/usr/bin/env bash
#
# Overnight batch: ingest each URL in URLS array, commit + push to web
# repo's main after every success so partial progress isn't lost.
#
# Usage: ./scripts/overnight.sh
# Env: HF_TOKEN must be set (in pipeline/.env)

set -uo pipefail

cd "$(dirname "$0")/.."
PIPELINE_DIR="$(pwd)"
WEB_DIR="$(cd ../web && pwd)"

# shellcheck disable=SC1091
source .venv/bin/activate

# Format: URL | group_slug | comma-separated tags | sensitivity
URLS=(
  "https://www.youtube.com/watch?v=gsg50GXGnQI|nakagawake|寄席,2024,保険|normal"
  "https://www.youtube.com/watch?v=Xg5hR1b8buY|nakagawake|寄席,人間ドック|normal"
  "https://www.youtube.com/watch?v=OuaztBkaSP8|nakagawake|寄席,2025,となりのマダム|normal"
  "https://www.youtube.com/watch?v=FWXlwdgyXY4|nakagawake|寄席,2024,デパート|normal"
)

ok=0
fail=0
fails=()

for entry in "${URLS[@]}"; do
    IFS='|' read -r url group tags sens <<< "$entry"
    echo
    echo "════════════════════════════════════════════════════════"
    echo "  $url  (group=$group sens=$sens tags=$tags)"
    echo "════════════════════════════════════════════════════════"

    # Build --tag args
    tag_args=()
    IFS=',' read -ra tag_arr <<< "$tags"
    for t in "${tag_arr[@]}"; do
        tag_args+=(--tag "$t")
    done

    if python -m pipeline ingest "$url" \
        --group-slug "$group" \
        --sensitivity "$sens" \
        "${tag_args[@]}"; then
        ok=$((ok+1))
        # Commit + push if new files exist
        cd "$WEB_DIR"
        if [ -n "$(git status --porcelain src/content/manzai)" ]; then
            git add src/content/manzai
            file=$(git diff --cached --name-only | head -1 | xargs basename .md)
            git commit -m "ingest: $file" \
                       -m "Auto-transcribed via pipeline (status: draft)." \
                       -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
            git push origin main
        else
            echo "  (no content change)"
        fi
        cd "$PIPELINE_DIR"
    else
        fail=$((fail+1))
        fails+=("$url")
    fi
done

echo
echo "════════════════════════════════════════════════════════"
echo "  done — $ok success, $fail failures"
if [ "$fail" -gt 0 ]; then
    printf '  failed: %s\n' "${fails[@]}"
fi
echo "════════════════════════════════════════════════════════"
