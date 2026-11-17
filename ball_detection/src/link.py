"""Turn per-frame candidates into one ball track by linking them through time.

Single-frame detections of a 5 px ball are full of false positives: heads, socks, pitch markings,
advertising. A real ball moves smoothly, so we build tracklets: each candidate either extends an
existing tracklet (within a gate that grows with the gap) or starts a new one. A tracklet's score
is the sum of its confidences; short or weak tracklets are dropped, and where tracklets overlap in
time the higher-scoring one wins.

    python ball_detection/src/link.py --name public
Output: data/ball/track_<name>.npz with per-frame arrays over the whole video:
    u, v (pixels), x, y (normalised pitch, ground-plane assumption), conf, linked (bool)
"""

import argparse
import os

import numpy as np

from common import BALL_DIR, PITCH, load_hom, px_to_pitch

# Params
GATE_PX = 30.0       # max movement per frame in full-res pixels (~1.5-2 m on this camera: a hard shot)
MAX_GAP = 12         # frames a tracklet may coast through without a detection
MIN_LEN = 6          # detections needed for a tracklet to count
MIN_SCORE = 2.0      # summed confidence needed for a tracklet to count
INTERP_GAP = 12      # fill holes inside an accepted tracklet up to this long
OFF_PITCH_M = 1.0    # candidates further outside the lines than this are ignored


def build_tracklets(cand):
    frames = np.unique(cand[:, 0]).astype(int)
    by_frame = {f: cand[cand[:, 0] == f] for f in frames}
    active, done = [], []
    for f in frames:
        dets = sorted(by_frame[f], key=lambda d: -d[5])
        used = set()
        # extend the strongest tracklets first
        for t in sorted(active, key=lambda t: -t["score"]):
            last_f, lu, lv = t["pts"][-1][:3]
            vu, vv = t["vel"]
            gap = f - last_f
            pu, pv = lu + vu * gap, lv + vv * gap
            best, bd = None, GATE_PX * gap
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = np.hypot(d[1] - pu, d[2] - pv)
                if dist < bd:
                    best, bd = i, dist
            if best is not None:
                d = dets[best]
                used.add(best)
                t["vel"] = (0.5 * t["vel"][0] + 0.5 * (d[1] - lu) / gap, 0.5 * t["vel"][1] + 0.5 * (d[2] - lv) / gap)
                t["pts"].append((f, d[1], d[2], d[5]))
                t["score"] += d[5]
        for i, d in enumerate(dets):
            if i not in used:
                active.append({"pts": [(f, d[1], d[2], d[5])], "vel": (0.0, 0.0), "score": d[5]})
        still = []
        for t in active:
            (still if f - t["pts"][-1][0] <= MAX_GAP else done).append(t)
        active = still
    return done + active


def select(tracklets, n_frames):
    u = np.full(n_frames, np.nan)
    v = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    owner = np.full(n_frames, -1.0)
    good = [t for t in tracklets if len(t["pts"]) >= MIN_LEN and t["score"] >= MIN_SCORE]
    for t in sorted(good, key=lambda t: t["score"]):  # strongest written last, so it wins overlaps
        pts = np.array(t["pts"])
        fs = pts[:, 0].astype(int)
        span = np.arange(fs[0], fs[-1] + 1)
        iu = np.interp(span, fs, pts[:, 1])
        iv = np.interp(span, fs, pts[:, 2])
        # only keep interpolated frames within INTERP_GAP of a real detection
        k = np.searchsorted(fs, span)
        prev_f = fs[np.clip(k - 1, 0, len(fs) - 1)]
        next_f = fs[np.clip(k, 0, len(fs) - 1)]
        hole = np.where(span == next_f, 0, next_f - prev_f)  # length of the hole each frame sits in
        ok = hole <= INTERP_GAP
        u[span[ok]], v[span[ok]] = iu[ok], iv[ok]
        conf[span[ok]] = np.interp(span, fs, pts[:, 3])[ok]
        owner[span[ok]] = t["score"]
    return u, v, conf, len(good)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--min-conf", type=float, default=0.05)
    args = p.parse_args()

    data = np.load(os.path.join(BALL_DIR, f"candidates_{args.name}.npz"))
    cand = data["cand"]
    H = load_hom()
    n = len(H)
    # Spare balls lie beside the pitch and form long, steady tracklets that would beat the match
    # ball. A ball more than OFF_PITCH_M outside the lines is out of play anyway, so drop it.
    px = np.empty(len(cand))
    py = np.empty(len(cand))
    for f in np.unique(cand[:, 0]).astype(int):
        m = cand[:, 0] == f
        px[m], py[m] = px_to_pitch(H[f], cand[m, 1], cand[m, 2])
    mx, my = OFF_PITCH_M / PITCH[0], OFF_PITCH_M / PITCH[1]
    on = (px > -mx) & (px < 1 + mx) & (py > -my) & (py < 1 + my) & (cand[:, 5] >= args.min_conf)
    print(f"{len(cand)} candidates, {on.sum()} on the pitch and >= conf {args.min_conf}")
    cand = cand[on]
    tracklets = build_tracklets(cand)
    u, v, conf, n_good = select(tracklets, n)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    for f in np.flatnonzero(np.isfinite(u)):
        x[f], y[f] = [c.item() for c in px_to_pitch(H[f], u[f], v[f])]
    out = os.path.join(BALL_DIR, f"track_{args.name}.npz")
    np.savez_compressed(out, u=u, v=v, x=x, y=y, conf=conf, linked=np.isfinite(u), ranges=data["ranges"])
    print(f"{len(tracklets)} tracklets, {n_good} kept, ball on {np.isfinite(u).sum()} frames -> {out}")
