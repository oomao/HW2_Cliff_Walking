#!/usr/bin/env bash
# dev:start — pull latest code, read the handover note, and open the current openspec change.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[startup] git pull --ff-only"
git pull --ff-only || echo "[startup] (no upstream or nothing to pull)"

HANDOVER="docs/HANDOVER.md"
if [ -f "$HANDOVER" ]; then
  echo ""
  echo "===== $HANDOVER ====="
  cat "$HANDOVER"
  echo "====================="
fi

if command -v openspec >/dev/null 2>&1; then
  echo ""
  echo "[startup] openspec list"
  openspec list || true
  echo ""
  echo "[startup] Active changes:"
  ls openspec/changes 2>/dev/null | grep -v '^archive$' || echo "(none)"
else
  echo "[startup] openspec CLI not found — skipping"
fi

echo ""
echo "[startup] Suggested next actions:"
echo "  1. Review the handover above (if any)."
echo "  2. Inspect active openspec changes under openspec/changes/."
echo "  3. Run tests/training:  PYTHONPATH=src python -m cliff_walking.train"
