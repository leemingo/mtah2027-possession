"""Inputs: tracking array, slot teams, Smart Tagging, graded halves, ball sources.

Everything is indexed by tactical video frame (0-based), i.e. by submission row.
Positions are returned in metres (pitch from games.py).
"""

import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "starter")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require  # noqa: E402
import baseline as _baseline  # noqa: E402  (starter/baseline.py, read-only use of its XML parser)
import load_smart_tagging  # noqa: E402

PITCH = GAMES[GAME]["pitch"]            # (105.3, 68.0) metres
FPS = GAMES[GAME]["fps"]                # 25
N_FRAMES = GAMES[GAME]["n_video_frames"]
HOME, AWAY = 0, 1
TEAM_OF_TAG = {"barcelona": HOME, "atletic club": AWAY}
CACHE_PATH = os.path.join(DATA_DIR, "cache", "tactical_atd.npy")


def load_tracking():
    """(N_FRAMES, n_slots + 1, 3) float32: x, y (0-1) and speed (m/s); last object is the ball.

    Same layout and cache file as dashboard.py. Built from the raw ATD on first use.
    """
    if os.path.isfile(CACHE_PATH):
        return np.load(CACHE_PATH)
    raw = require(["atd_raw"])[0]
    with open(raw, "rb") as handle:
        text = handle.read().replace(b":", b",").replace(b";", b",")
    table = pd.read_csv(io.BytesIO(text), header=None, dtype=np.float32, engine="c").values
    frame_ids = table[:, 0].astype(np.int64)
    objects = table[:, 1:].reshape(len(table), -1, 3)
    out = np.full((N_FRAMES,) + objects.shape[1:], np.nan, dtype=np.float32)
    out[frame_ids - 1] = objects  # ATD frame_id counts from 1
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.save(CACHE_PATH, out)
    return out


def slot_teams():
    """int8 array over the 46 track slots: 0 home (Barcelona), 1 away."""
    return _baseline.team_of_each_track(require(["atd_meta"])[0])


def graded_blocks():
    """[(half, first, last)] inclusive video-frame ranges of the two graded halves."""
    return half_video_blocks(GAME)


def smart_tags():
    """Smart Tagging instances with video-frame intervals (start inclusive, end exclusive)."""
    return load_smart_tagging.load()


def players_and_ball(trk):
    """(players_xy_m (n,46,2), players_speed (n,46), ball_xy_m (n,2)) as float64."""
    scale = np.array(PITCH, dtype=np.float64)
    players = trk[:, :-1, :2].astype(np.float64) * scale
    speed = trk[:, :-1, 2].astype(np.float64)
    ball = trk[:, -1, :2].astype(np.float64) * scale
    return players, speed, ball


def run_filter(mask, min_len):
    """Keep only the True runs of mask that are at least min_len frames long."""
    mask = np.asarray(mask, dtype=bool)
    if min_len <= 1:
        return mask.copy()
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    out = np.zeros_like(mask)
    for start, stop in zip(edges[0::2], edges[1::2]):
        if stop - start >= min_len:
            out[start:stop] = True
    return out


def aux_ball(npz_path, key, conf_min, min_len):
    """Auxiliary ball track (metres) from a ball-detection bundle, gated by sustained confidence.

    A frame is kept only inside a run of >= min_len consecutive frames with conf >= conf_min.
    Returns (xy (n,2) with NaN where not kept, kept mask).
    """
    bundle = np.load(npz_path)
    # Either a viewer bundle (<key>__x, ...) or a ball_detection track file (x, y, conf).
    prefix = key + "__" if key + "__x" in bundle.files else ""
    x = bundle[prefix + "x"].astype(np.float64)
    y = bundle[prefix + "y"].astype(np.float64)
    conf = np.nan_to_num(bundle[prefix + "conf"].astype(np.float64), nan=0.0)
    present = np.isfinite(x) & np.isfinite(y)
    kept = run_filter(present & (conf >= conf_min), min_len)
    xy = np.full((len(x), 2), np.nan)
    xy[kept, 0] = x[kept] * PITCH[0]
    xy[kept, 1] = y[kept] * PITCH[1]
    return xy, kept


def ball_source(trk, aux=None):
    """(ball_xy_m (n,2), source (n,) int8: 0 none, 1 Metrica, 2 auxiliary).

    The auxiliary track only fills frames where Metrica has no ball.
    """
    _, _, ball = players_and_ball(trk)
    source = np.where(np.isfinite(ball[:, 0]), 1, 0).astype(np.int8)
    if aux is not None:
        aux_xy, kept = aux
        fill = (source == 0) & kept
        ball = ball.copy()
        ball[fill] = aux_xy[fill]
        source[fill] = 2
    return ball, source
