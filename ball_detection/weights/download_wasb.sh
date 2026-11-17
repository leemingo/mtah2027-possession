#!/usr/bin/env bash
# Downloads the public WASB soccer weights (zero-shot detector) from the WASB model zoo
# (https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md). Not committed: third-party file.
set -euo pipefail
cd "$(dirname "$0")"
[ -f wasb_soccer_best.pth.tar ] || gdown 1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l -O wasb_soccer_best.pth.tar
