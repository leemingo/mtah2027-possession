# Ball detection (WASB zero-shot)

Ball detection used by the submitted variants (P0B05, LP0B05):
WASB (Tarashima et al., BMVC 2023, https://github.com/nttcom/WASB-SBDT, MIT) with its **public
soccer weights**, run zero-shot on the tactical video, then linked into one ball track.

| file | role |
|---|---|
| `weights/download_wasb.sh` | downloads the public WASB soccer weights (model zoo, via `gdown`) |
| `src/wasb_detect.py` | WASB HRNet over the video in crops (zoom 1.5) → `data/ball/candidates_<name>.npz` |
| `src/link.py` | links candidates into tracklets and keeps the best → `data/ball/track_<name>.npz` |
| `src/common.py` | paths, video streaming, homography helpers |
| `external/WASB-SBDT/` | vendored WASB source (MIT, upstream commit 923462c) |

```bash
bash ball_detection/weights/download_wasb.sh
python ball_detection/src/wasb_detect.py --halves --zoom 1.5 --name wasb_full   # GPU, ~1.5 h per half
python ball_detection/src/link.py --name wasb_full                              # ~10 min
```

Outputs are frame-level derived data: they stay in `data/ball/` (gitignored) and must not be
published.
