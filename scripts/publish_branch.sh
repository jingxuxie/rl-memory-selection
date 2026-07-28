#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jingxuxie/rl-memory-selection.git}"
BRANCH="${BRANCH:-agent/decision-relevant-memory}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone "$REPO_URL" "$TMP/repo"
if git -C "$TMP/repo" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git -C "$TMP/repo" checkout -B "$BRANCH" "origin/$BRANCH"
else
  git -C "$TMP/repo" checkout -B "$BRANCH" origin/main
fi

# Remove the obsolete packed-bootstrap handoff while preserving the official
# author kit inherited from main.
rm -rf "$TMP/repo/bootstrap"
rm -f "$TMP/repo/.github/workflows/bootstrap-drms.yml" "$TMP/repo/paper/.gitkeep"

rsync -a \
  --exclude '.git/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'submission/' \
  "$ROOT/" "$TMP/repo/"

git -C "$TMP/repo" add -A
if git -C "$TMP/repo" diff --cached --quiet; then
  echo "No changes to publish."
  exit 0
fi

git -C "$TMP/repo" commit -m "Expand DRMS theory, experiments, and AAAI manuscript"
git -C "$TMP/repo" push -u origin "$BRANCH"
echo "Published $BRANCH to $REPO_URL"
