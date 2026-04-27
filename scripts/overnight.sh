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

# Format: URL | group_slug | comma-separated tags | sensitivity | language(opt)
# Each must be a COMPLETE performance/act (no short clips, no compilations).
URLS=(
  # 中川家 official channel — full 寄席 acts
  "https://www.youtube.com/watch?v=gsg50GXGnQI|nakagawake|寄席,2024,保険|normal|"
  "https://www.youtube.com/watch?v=Xg5hR1b8buY|nakagawake|寄席,人間ドック|normal|"
  "https://www.youtube.com/watch?v=OuaztBkaSP8|nakagawake|寄席,2025,となりのマダム|normal|"
  "https://www.youtube.com/watch?v=FWXlwdgyXY4|nakagawake|寄席,2024,デパート|normal|"
  "https://www.youtube.com/watch?v=a88gqadWS-Y|nakagawake|寄席,2023,おばちゃんタクシー|normal|"
  # 漫才兄弟 (徐浩伦 + 谭湘文) — Chinese manzai, full acts
  "https://www.youtube.com/watch?v=VVVqPKkS9zc|manzai-brothers|央视春晚,2025,骗假不留|normal|zh"
  "https://www.youtube.com/watch?v=_Fw3jXi5Eyk|manzai-brothers|湖南春晚,2025,理发师|normal|zh"
  "https://www.youtube.com/watch?v=vKLVy2p12tA|manzai-brothers|脱口秀,2024,总决赛,柯南|high|zh"
)

ok=0
fail=0
fails=()

for entry in "${URLS[@]}"; do
    IFS='|' read -r url group tags sens lang <<< "$entry"
    echo
    echo "════════════════════════════════════════════════════════"
    echo "  $url"
    echo "  group=$group sens=$sens tags=$tags lang=${lang:-auto}"
    echo "════════════════════════════════════════════════════════"

    # Build --tag args
    tag_args=()
    IFS=',' read -ra tag_arr <<< "$tags"
    for t in "${tag_arr[@]}"; do
        tag_args+=(--tag "$t")
    done

    # Optional --language
    lang_args=()
    if [ -n "$lang" ]; then
        lang_args=(--language "$lang")
    fi

    if python -m pipeline ingest "$url" \
        --group-slug "$group" \
        --sensitivity "$sens" \
        "${tag_args[@]}" \
        "${lang_args[@]}"; then
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
