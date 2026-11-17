"""Shared helpers for the ball-detection pipeline: data access, geometry, tiling, video streaming.

The tactical ball is ~4-5 px wide at 1920x1080, far below what a detector sees at its native
input size. Every stage therefore works on *upscaled tiles*: the frame is resized by SCALE and cut
into TILE x TILE crops, so at SCALE=2 the ball is ~8-10 px inside a 640 px tile.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # ball_detection/src
PKG = os.path.dirname(HERE)                                 # ball_detection/
ROOT = os.path.dirname(PKG)                                 # repository root (games.py, starter/)
sys.path.insert(0, ROOT)
sys.path.insert(0, PKG)                                     # dashboard.py builds the caches on first use
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache")
# All outputs of this pipeline (gitignored). MTAH_BALL_DIR moves them, e.g. to /kaggle/working
# when data/ itself is a read-only Kaggle input.
BALL_DIR = os.environ.get("MTAH_BALL_DIR", os.path.join(DATA_DIR, "ball"))
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")  # public datasets
IMAGE_SIZE = (1920, 1080)
TILE = 640
OVERLAP = 96
SCALE = 2.0
PITCH = GAMES[GAME]["pitch"]
# Public 5 minutes with possession GT (frame range).
HOLDOUT = (11185, 18684)


def load_atd():
    """(n_frames, n_slots + 1, 3), last object is the ball. Built by dashboard.py on first run."""
    path = os.path.join(CACHE_DIR, "tactical_atd.npy")
    if not os.path.isfile(path):
        import dashboard  # noqa: F401  (heavy import, only needed once)
        dashboard.load_tracking("tactical")
    return np.load(path, mmap_mode="r")


def load_hom():
    path = os.path.join(CACHE_DIR, "tactical_hom.npy")
    if not os.path.isfile(path):
        import dashboard
        dashboard.load_homography_arrays("tactical")
    return np.load(path, mmap_mode="r")


def pitch_to_px(H, x, y):
    """Normalised pitch (0-1) -> pixels. Vectorised over x, y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = H @ np.stack([x, y, np.ones_like(x)])
    return p[0] / p[2], p[1] / p[2]


def px_to_pitch(H, u, v):
    """Pixels -> normalised pitch, assuming the point is on the ground (wrong for a ball in the air)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    p = np.linalg.inv(H) @ np.stack([u, v, np.ones_like(u)])
    return p[0] / p[2], p[1] / p[2]


def metrica_ball_px(atd, H, frames):
    """(n, 2) projected Metrica ball pixel positions, NaN where Metrica has no ball."""
    out = np.full((len(frames), 2), np.nan)
    for i, f in enumerate(frames):
        bx, by = atd[f, -1, :2]
        if np.isfinite(bx):
            out[i] = [c.item() for c in pitch_to_px(H[f], bx, by)]
    return out


def pitch_mask_tiles(H, scale=SCALE, margin_m=3.0):
    """Tile origins (x0, y0) in the upscaled frame that overlap the pitch (+margin).

    The tactical camera spends the top of the frame on stands; skipping those tiles saves ~30%.
    """
    w, h = int(IMAGE_SIZE[0] * scale), int(IMAGE_SIZE[1] * scale)
    mx, my = margin_m / PITCH[0], margin_m / PITCH[1]
    xs = np.linspace(-mx, 1 + mx, 40)
    ys = np.linspace(-my, 1 + my, 26)
    gx, gy = np.meshgrid(xs, ys)
    u, v = pitch_to_px(H, gx.ravel(), gy.ravel())
    u, v = u * scale, v * scale
    step = TILE - OVERLAP
    xs0 = sorted(set(list(range(0, w - TILE, step)) + [w - TILE]))
    ys0 = sorted(set(list(range(0, h - TILE, step)) + [h - TILE]))
    keep = []
    for y0 in ys0:
        for x0 in xs0:
            inside = (u >= x0) & (u < x0 + TILE) & (v >= y0) & (v < y0 + TILE)
            if inside.any():
                keep.append((x0, y0))
    return keep


def cut_tiles(img, origins, scale=SCALE):
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else img
    return [big[y0:y0 + TILE, x0:x0 + TILE] for x0, y0 in origins]


def stream_frames(start, end, step=1):
    """Yield (frame_index, BGR image) for start..end inclusive. Seeks once, then decodes linearly."""
    cap = cv2.VideoCapture(asset_paths()["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    f = start
    while f <= end:
        ok = cap.grab()
        if not ok:
            break
        if (f - start) % step == 0:
            ok, img = cap.retrieve()
            if ok:
                yield f, img
        f += 1
    cap.release()


def graded_frames():
    return [(a, b) for _, a, b in half_video_blocks(GAME)]
