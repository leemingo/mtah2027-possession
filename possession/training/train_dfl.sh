#!/usr/bin/env bash
# Download Sportec Open DFL (Bassek et al., Scientific Data 2025; CC BY 4.0) match information and
# raw positions from figshare article 28196177, parse each match into training arrays, and
# train the learned emission. Run from the repository root; ~2.6 GB of XML.
#   PY_DFL: Python with possession/training/requirements-dfl.txt   PY_POSS: requirements-possession.txt
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=${DFL_RAW:-external/dfl/raw}; CACHE=${DFL_CACHE:-external/dfl/cache}
mkdir -p "$RAW" "$CACHE"
while read -r name url md5; do
  [ -f "$RAW/$name" ] || curl -sSL -o "$RAW/$name" "$url"
  echo "$md5  $RAW/$name" | md5sum -c --quiet
done <<'LIST'
DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml https://ndownloader.figshare.com/files/51643472 37b0f85588c0acfb995fe633c7ac2e0d
DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml https://ndownloader.figshare.com/files/51643475 bc5d5b21dbcf36aa30f4cd0428309f6b
DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml https://ndownloader.figshare.com/files/51643478 b9710781d400dd812f62b0a6e3d7bce8
DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml https://ndownloader.figshare.com/files/51643490 5e2d580d3d8ed97e32959de90d04992b
DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml https://ndownloader.figshare.com/files/51643487 65a4568e4f374af94d3c1015cbc7c4f5
DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml https://ndownloader.figshare.com/files/51643484 5979209af316a914ce57347d0633df6c
DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml https://ndownloader.figshare.com/files/51643481 7b021e577fca917a63111d55cc29ae5b
DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml https://ndownloader.figshare.com/files/51643514 e4273a2adf5dbd9d4ec42def2f01905c
DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml https://ndownloader.figshare.com/files/51643517 bacb2f609c1d1b0bfc906f6b9d4109f2
DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml https://ndownloader.figshare.com/files/51643520 3efff25a86a61768e7d253b4bba137a3
DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml https://ndownloader.figshare.com/files/51643523 4e1063e0a0ae499920cf68ac1aa3ab4e
DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml https://ndownloader.figshare.com/files/51643526 e56ff1336a334f8da51110fbe7cab764
DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml https://ndownloader.figshare.com/files/51643529 9105bad987191d3c3c362c1fce9a4fcd
DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml https://ndownloader.figshare.com/files/51643532 e060bb7d7a5a3bfbd15bbf9bc631024d
LIST
for meta in "$RAW"/DFL_02_01_matchinformation_*.xml; do
  id=${meta##*DFL-MAT-}; id=${id%.xml}
  pos=$(ls "$RAW"/DFL_04_03_positions_raw_observed_*"$id".xml)
  [ -f "$CACHE/$id.npz" ] || "${PY_DFL:?set PY_DFL}" -I possession/training/dfl_parse.py "$meta" "$pos" "$CACHE/$id.npz"
done
OMP_NUM_THREADS=16 "${PY_POSS:-python}" -m possession.learned train --cache "$CACHE" --out possession/models/learned_dfl.pkl
md5sum possession/models/learned_dfl.pkl
