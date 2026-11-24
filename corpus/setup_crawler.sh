#!/usr/bin/env bash
# Set up edgar-crawler exactly as it was used to build the corpus:
#   1. clone it at the pinned commit
#   2. apply our local patch (rate limiter: <= 8 requests/s, the SEC fair-access limit)
#   3. write config.json from the template, with SEC_USER_AGENT from .env
# Safe to re-run. Override the target with CORPUS_DIR=... (default: data/edgar-software-corpus).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS_DIR="${CORPUS_DIR:-$REPO/data/edgar-software-corpus}"
CRAWLER="$CORPUS_DIR/edgar-crawler"
URL="https://github.com/nlpaueb/edgar-crawler"
COMMIT="84a8d0c5dd7dd6769526e5ccec534c4e0880d56f"   # 2025-07-18, the version the corpus was built with
PATCH="$REPO/corpus/edgar-crawler-throttle.patch"

# SEC_USER_AGENT comes from .env (format: "Your Name your@email.com")
[ -f "$REPO/.env" ] && set -a && . "$REPO/.env" && set +a
: "${SEC_USER_AGENT:?Set SEC_USER_AGENT in .env (see .env.example)}"

mkdir -p "$CORPUS_DIR/scripts"
if [ ! -d "$CRAWLER/.git" ]; then
  git clone -q "$URL" "$CRAWLER"
  git -C "$CRAWLER" checkout -q "$COMMIT"
fi
echo "edgar-crawler at $(git -C "$CRAWLER" rev-parse --short HEAD) (expected ${COMMIT:0:7})"

if git -C "$CRAWLER" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "throttle patch: already applied"
else
  git -C "$CRAWLER" apply "$PATCH" && echo "throttle patch: applied"
fi

# Fill the user agent into the config template (via Python so quoting is always valid JSON).
"$REPO/.venv/bin/python" - "$REPO/corpus/crawler_config.template.json" "$CRAWLER/config.json" <<'EOF'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
cfg["download_filings"]["user_agent"] = os.environ["SEC_USER_AGENT"]
json.dump(cfg, open(sys.argv[2], "w"), indent="\t")
print(f"wrote {sys.argv[2]}")
EOF
