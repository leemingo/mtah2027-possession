#!/usr/bin/env bash
# Fetch PathCRF (https://github.com/hyunsungkim-ds/pathcrf, MPL-2.0) at the pinned commit into
# external/pathcrf. The pretrained checkpoint saved/140 ships inside that repository.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=1b7c6f85a6b78c92e4543fde72ca2b86a933f997
DEST=external/pathcrf
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/hyunsungkim-ds/pathcrf.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" || true
git -C "$DEST" checkout --quiet "$COMMIT"
echo "PathCRF at $(git -C "$DEST" rev-parse HEAD)"
ls -l "$DEST/saved/140/model/state_dict_best_acc.pt"
