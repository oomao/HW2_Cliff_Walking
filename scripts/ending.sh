#!/usr/bin/env bash
# dev:ending — update tasks, archive the current change if complete, write the
# handover note, and push to github.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v openspec >/dev/null 2>&1; then
  echo "[ending] openspec validate (non-strict, all active changes)"
  for d in openspec/changes/*/; do
    name="$(basename "$d")"
    [ "$name" = "archive" ] && continue
    echo "  -> $name"
    openspec validate "$name" || true
  done
fi

HANDOVER="docs/HANDOVER.md"
mkdir -p docs
{
  echo "# Handover"
  echo ""
  echo "_Last updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)_"
  echo ""
  echo "## Current branch"
  echo ""
  echo "- $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no git)')"
  echo ""
  echo "## Active openspec changes"
  echo ""
  ls openspec/changes 2>/dev/null | grep -v '^archive$' | sed 's/^/- /' || echo "- (none)"
  echo ""
  echo "## Next actions"
  echo ""
  echo "- Run \`scripts/startup.sh\` to resume."
  echo "- See README.md for reproducing figures."
} > "$HANDOVER"

echo "[ending] handover written to $HANDOVER"

if [ -n "${PUSH:-}" ]; then
  echo "[ending] git add / commit / push"
  git add -A
  git commit -m "dev:ending checkpoint" || echo "(nothing to commit)"
  git push
else
  echo "[ending] set PUSH=1 to commit & push automatically"
fi
