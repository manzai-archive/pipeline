#!/usr/bin/env bash
#
# Batch ingest via Docker container. Parallel ingest with serialized git push.
# Each URL: docker compose run -> ingest; after every ingest the host
# acquires a git lock and commits + pushes.
#
# Run from pipeline repo root on a Docker+GPU host:
#   ./scripts/batch-ingest.sh           # default parallelism
#   PARALLEL=2 ./scripts/batch-ingest.sh
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
LOCK="/tmp/manzai-git.lock"
PARALLEL="${PARALLEL:-3}"

# Format: URL | group_slug | comma-separated tags | sensitivity | language(opt)
URLS=(
  # 中川家 official channel — full 寄席 acts
  "https://www.youtube.com/watch?v=gsg50GXGnQI|nakagawake|寄席,2024,保険|normal|"
  "https://www.youtube.com/watch?v=Xg5hR1b8buY|nakagawake|寄席,人間ドック|normal|"
  "https://www.youtube.com/watch?v=OuaztBkaSP8|nakagawake|寄席,2025,となりのマダム|normal|"
  "https://www.youtube.com/watch?v=FWXlwdgyXY4|nakagawake|寄席,2024,デパート|normal|"
  "https://www.youtube.com/watch?v=a88gqadWS-Y|nakagawake|寄席,2023,おばちゃんタクシー|normal|"
  # 漫才兄弟 (徐浩伦 + 谭湘文) — Chinese manzai
  "https://www.youtube.com/watch?v=VVVqPKkS9zc|manzai-brothers|央视春晚,2025,骗假不留|normal|zh"
  "https://www.youtube.com/watch?v=_Fw3jXi5Eyk|manzai-brothers|湖南春晚,2025,理发师|normal|zh"
  "https://www.youtube.com/watch?v=STTxgW1QeqI|manzai-brothers|央视春晚,2026,谁的菜|normal|zh"
)

run_one() {
    local entry="$1"
    IFS='|' read -r url group tags sens lang <<< "$entry"
    echo "[start] $url ($group / ${tags:-})"

    tag_args=()
    IFS=',' read -ra tag_arr <<< "$tags"
    for t in "${tag_arr[@]}"; do
        tag_args+=(--tag "$t")
    done
    lang_args=()
    if [ -n "$lang" ]; then
        lang_args=(--language "$lang")
    fi

    local log="/tmp/ingest.${BASHPID:-$$}.log"
    if ! docker compose run --rm pipeline ingest "$url" \
        --group-slug "$group" \
        --sensitivity "$sens" \
        "${tag_args[@]}" \
        "${lang_args[@]}" >"$log" 2>&1; then
        echo "[FAIL] $url (see $log)"
        return 1
    fi

    # Serialized git commit + push
    (
        flock -x 9
        cd "$WEB_DIR"
        if [ -n "$(git status --porcelain src/content)" ]; then
            git add src/content
            file=$(git diff --cached --name-only | grep "src/content/manzai/" | head -1 | xargs -n1 basename | sed 's/\.md$//')
            git pull --rebase --quiet origin main 2>/dev/null || true
            git commit -q -m "ingest: $file" \
                       -m "Auto-transcribed via qwen-omni pipeline." \
                       -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" || true
            git push -q origin main || true
            echo "[OK] $url -> $file"
        else
            echo "[OK] $url (no content change)"
        fi
        cd "$PIPELINE_DIR"
    ) 9>"$LOCK"
}

export -f run_one
export PIPELINE_DIR WEB_DIR LOCK

# xargs -P for parallelism
printf "%s\n" "${URLS[@]}" \
    | xargs -d '\n' -I {} -P "$PARALLEL" \
        bash -c 'run_one "$@"' _ {}

echo
echo "================================================================"
echo "  batch done"
echo "================================================================"
