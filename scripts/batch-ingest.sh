#!/usr/bin/env bash
#
# Overnight batch via Docker container.
# Each URL: docker compose run -> ingest -> commit + push web repo.
#
# Run from pipeline repo root on a Docker+GPU host:
#   ./scripts/overnight-docker.sh
#
# Requires: docker compose build done; .env with HF_TOKEN; ../web cloned;
#           git push to ../web works (gh auth or SSH key).

set -o pipefail

# Proxy for `git push` and other host-side network ops. The container
# gets its own proxy via .env. Set GIT_PROXY=http://... in your shell
# to override; leave unset on hosts with direct internet.
GIT_PROXY="${GIT_PROXY:-${HTTPS_PROXY:-}}"
if [ -n "$GIT_PROXY" ]; then
    export HTTPS_PROXY="$GIT_PROXY"
    export HTTP_PROXY="$GIT_PROXY"
fi

cd "$(dirname "$0")/.."
PIPELINE_DIR="$(pwd)"
WEB_DIR="$(cd ../web && pwd)"

# Format: URL | group_slug | comma-separated tags | sensitivity | language(opt)
URLS=(
  # 中川家 official channel — full 寄席 acts
  "https://www.youtube.com/watch?v=gsg50GXGnQI|nakagawake|寄席,2024,保険|normal|"
  "https://www.youtube.com/watch?v=Xg5hR1b8buY|nakagawake|寄席,人間ドック|normal|"
  "https://www.youtube.com/watch?v=OuaztBkaSP8|nakagawake|寄席,2025,となりのマダム|normal|"
  "https://www.youtube.com/watch?v=FWXlwdgyXY4|nakagawake|寄席,2024,デパート|normal|"
  "https://www.youtube.com/watch?v=a88gqadWS-Y|nakagawake|寄席,2023,おばちゃんタクシー|normal|"
  # 漫才兄弟 (徐浩伦 + 谭湘文) — Chinese manzai, single broadcast acts only
  "https://www.youtube.com/watch?v=VVVqPKkS9zc|manzai-brothers|央视春晚,2025,骗假不留|normal|zh"
  "https://www.youtube.com/watch?v=_Fw3jXi5Eyk|manzai-brothers|湖南春晚,2025,理发师|normal|zh"
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

    tag_args=()
    IFS=',' read -ra tag_arr <<< "$tags"
    for t in "${tag_arr[@]}"; do
        tag_args+=(--tag "$t")
    done

    lang_args=()
    if [ -n "$lang" ]; then
        lang_args=(--language "$lang")
    fi

    if docker compose run --rm pipeline ingest "$url" \
        --group-slug "$group" \
        --sensitivity "$sens" \
        "${tag_args[@]}" \
        "${lang_args[@]}"; then

        ok=$((ok+1))
        cd "$WEB_DIR"
        if [ -n "$(git status --porcelain src/content)" ]; then
            git add src/content
            file=$(git diff --cached --name-only | grep "src/content/manzai/" | head -1 | xargs -n1 basename | sed 's/\.md$//')
            git commit -m "ingest: $file" \
                       -m "Auto-transcribed via pipeline (status: draft) on GPU host." \
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
