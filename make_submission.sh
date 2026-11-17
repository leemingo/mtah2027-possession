#!/usr/bin/env bash
# Regenerate a possession submission from the provided data (see DATA.md for data/).
#
#   bash make_submission.sh A     OUT.csv   # tracking + Smart Tagging, numpy/pandas only
#   bash make_submission.sh P0    OUT.csv   # + PathCRF ball-free team prior (CPU torch)
#   bash make_submission.sh P0B05 OUT.csv   # + video ball (WASB zero-shot, GPU) where Metrica has none
#   bash make_submission.sh LP0B05 OUT.csv  # + learned emission (classifier trained on Sportec Open DFL,
#                                           #   possession/models/learned_dfl.pkl; retrain with
#                                           #   possession/training/train_dfl.sh)
#
# Interpreters (each from its own environment, see README "Reproduction"):
#   PY_POSS  requirements-possession.txt   (Python 3.10)          default: python
#   PY_PCRF  requirements-pathcrf.txt      (Python 3.11, CPU torch)
#   PY_BALL  ball_detection/requirements.txt (Python 3.12, CUDA torch)
set -euo pipefail
cd "$(dirname "$0")"
VARIANT=${1:?variant: A | P0 | P0B05 | LP0B05}
OUT=${2:?output csv}
PY_POSS=${PY_POSS:-python}
CACHE=${MTAH_DATA_DIR:-data}/cache
BALL_DIR=${MTAH_BALL_DIR:-${MTAH_DATA_DIR:-data}/ball}
ARGS=()

if [[ "$VARIANT" == *P0* ]]; then
  bash possession/setup_pathcrf.sh
  "${PY_PCRF:?set PY_PCRF}" -m possession.pathcrf_adapter --repo external/pathcrf \
      --out "$CACHE/pathcrf_p_home.npy"
  ARGS+=(--pathcrf "$CACHE/pathcrf_p_home.npy")
fi

if [[ "$VARIANT" == *B05 ]]; then
  PB=${PY_BALL:?set PY_BALL}
  PATH="$(dirname "$PB"):$PATH" bash ball_detection/weights/download_wasb.sh
  "$PB" ball_detection/src/wasb_detect.py --halves --zoom 1.5 --name wasb_full   # zero-shot, public weights
  "$PB" ball_detection/src/link.py --name wasb_full
  ARGS+=(--aux "$BALL_DIR/track_wasb_full.npz" --aux-key wasb_full --set AUX_CONF=0.5)
fi

if [[ "$VARIANT" == L* ]]; then
  # Learned evidence is ~2.3x stronger per frame than the rules on sustained-control frames, so all
  # three transition costs are scaled 12 -> 27.5 to keep "about 0.5 s of clean control" per change.
  ARGS+=(--model possession/models/learned_dfl.pkl --set W_L=0.5
         --set C_SWITCH=27.5 --set C_DEAD=27.5 --set C_RESTART=27.5)
fi

"$PY_POSS" -m possession.run --out "$OUT" "${ARGS[@]}"
"$PY_POSS" evaluation/validate_submission.py "$OUT"
md5sum "$OUT"
