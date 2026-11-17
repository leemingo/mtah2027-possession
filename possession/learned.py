"""Learned frame evidence: a gradient-boosted classifier P(H, A, D | frame features), trained on
OTHER matches with ground-truth labels (Sportec Open DFL, optionally SkillCorner open data), used
as the per-frame emission of the same HMM.

Features are team-symmetric geometry and kinematics around the ball plus collective motion and
temporal context; missing values are NaN (histogram gradient boosting handles them natively), so
the model also learns what a ball gap looks like. Nothing from the competition match is used for
training.

    python -m possession.learned train --cache DIR [--cache DIR2 ...] --out model.pkl
"""

import argparse
import glob
import os
import pickle

import numpy as np

from . import data
from .evidence import ball_speed, _centred_mean, _row_nanmean

H, A, D = 0, 1, 2
CLASSES = np.array(["H", "A", "D"])
FPS = data.FPS


def _nan_min2(d):
    """Smallest and second smallest along axis 1, NaN-aware (NaN where not enough values)."""
    s = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
    m1 = s[:, 0] if s.shape[1] else np.full(len(d), np.inf)
    m2 = s[:, 1] if s.shape[1] > 1 else np.full(len(d), np.inf)
    return np.where(np.isfinite(m1), m1, np.nan), np.where(np.isfinite(m2), m2, np.nan)


def _gap_context(present, cap=250):
    """Frames since the ball was last seen and until it is seen next (capped)."""
    n = len(present)
    idx = np.arange(n)
    last = np.where(present, idx, -10 ** 9)
    last = np.maximum.accumulate(last)
    nxt = np.where(present, idx, 10 ** 9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return np.minimum(idx - last, cap).astype(np.float32), np.minimum(nxt - idx, cap).astype(np.float32)


def features(trk, team):
    """(n, F) float32 feature matrix for every frame. trk as in data.load_tracking()."""
    players, speed, ball = data.players_and_ball(trk)
    L, W = data.PITCH
    n = len(trk)
    present = np.isfinite(ball[:, 0])
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    home, away = team == data.HOME, team == data.AWAY
    dh1, dh2 = _nan_min2(d[:, home])
    da1, da2 = _nan_min2(d[:, away])
    with np.errstate(invalid="ignore"):
        nh2 = np.where(present, np.nansum(d[:, home] < 2.0, axis=1), np.nan)
        na2 = np.where(present, np.nansum(d[:, away] < 2.0, axis=1), np.nan)
        nh5 = np.where(present, np.nansum(d[:, home] < 5.0, axis=1), np.nan)
        na5 = np.where(present, np.nansum(d[:, away] < 5.0, axis=1), np.nan)
    dd = np.where(np.isfinite(d), d, np.inf)
    nearest = np.argmin(dd, axis=1)
    near_speed = np.where(present, speed[np.arange(n), nearest], np.nan)
    vb = ball_speed(ball)
    v5 = np.full(n, np.nan)
    diff = ball[10:] - ball[:-10]
    v5[5:-5] = np.hypot(diff[:, 0], diff[:, 1]) * FPS / 10.0
    # direction change (touch indicator): angle between incoming and outgoing displacement
    turn = np.full(n, np.nan)
    a_in = ball[3:-3] - ball[:-6]
    a_out = ball[6:] - ball[3:-3]
    num = (a_in * a_out).sum(1)
    den = np.hypot(a_in[:, 0], a_in[:, 1]) * np.hypot(a_out[:, 0], a_out[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        turn[3:-3] = np.where(den > 1e-6, num / den, np.nan)
    bx, by = ball[:, 0], ball[:, 1]
    goal_dist = np.minimum(bx, L - bx)
    with np.errstate(invalid="ignore"):
        outside = np.maximum.reduce([-bx, bx - L, -by, by - W, np.zeros(n)])
        outside = np.where(present, outside, np.nan)
    ddiff = da1 - dh1
    mean_speed = _row_nanmean(speed)
    sp_h = _row_nanmean(np.where(home[None, :], speed, np.nan))
    sp_a = _row_nanmean(np.where(away[None, :], speed, np.nan))
    cnt_h = np.isfinite(players[:, home, 0]).sum(1).astype(np.float32)
    cnt_a = np.isfinite(players[:, away, 0]).sum(1).astype(np.float32)
    with np.errstate(invalid="ignore"):
        ctrl_h = np.where(present, (dh1 < 1.5) & ~(da1 < 1.5), np.nan)
        ctrl_a = np.where(present, (da1 < 1.5) & ~(dh1 < 1.5), np.nan)
    since, until = _gap_context(present)
    cols = [
        present.astype(np.float32), dh1, da1, dh2, da2, ddiff, nh2, na2, nh5, na5,
        vb, v5, turn, near_speed, np.abs(vb - near_speed),
        goal_dist, np.abs(by - W / 2), outside,
        mean_speed, _centred_mean(mean_speed, 51), _centred_mean(mean_speed, 251),
        _centred_mean(sp_h, 51), _centred_mean(sp_a, 51), cnt_h, cnt_a,
        _centred_mean(ddiff, 13), _centred_mean(ddiff, 51), _centred_mean(ddiff, 125),
        _centred_mean(ctrl_h, 25), _centred_mean(ctrl_a, 25),
        _centred_mean(ctrl_h, 125), _centred_mean(ctrl_a, 125),
        _centred_mean(vb, 25), _centred_mean(present.astype(np.float64), 125),
        since, until,
    ]
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)


FEATURE_NAMES = [
    "ball_present", "d_home", "d_away", "d2_home", "d2_away", "d_diff", "n_home_2m", "n_away_2m",
    "n_home_5m", "n_away_5m", "ball_speed", "ball_speed_10f", "ball_turn_cos", "near_speed",
    "rel_speed", "goal_dist", "abs_y_centre", "outside_dist", "mean_speed", "mean_speed_2s",
    "mean_speed_10s", "home_speed_2s", "away_speed_2s", "n_home", "n_away", "d_diff_0.5s",
    "d_diff_2s", "d_diff_5s", "ctrl_home_1s", "ctrl_away_1s", "ctrl_home_5s", "ctrl_away_5s",
    "ball_speed_1s", "ball_seen_5s", "frames_since_ball", "frames_until_ball",
]


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def predict_proba(model, X):
    """(n, 3) probabilities ordered H, A, D."""
    P = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return P[:, order]


def learned_evidence(P, p):
    """Tempered log-probabilities as additive scores, max-normalised per frame."""
    q = (1 - p["L_EPS"]) * P + p["L_EPS"] / 3.0
    S = p["W_L"] * np.log(q)
    return S - S.max(axis=1, keepdims=True)


def training_rows(trk, team, gt, step):
    X = features(trk, team)
    keep = np.flatnonzero(np.isin(gt, CLASSES))
    keep = keep[::step]
    return X[keep], gt[keep]


# Degradations that make the training matches look like ours: the ball removed on 41 % of the
# frames in long blocks (lognormal lengths, median 268 frames as measured on our match), and
# ~0.8 player per team missing in random stretches.
GAP_MEDIAN, GAP_SIGMA, BALL_MISSING = 268.0, 0.948, 0.41
PL_MEDIAN, PL_SIGMA, PL_MISSING = 100.0, 1.0, 0.8 / 11.0
TRAIN_VERSIONS = (("clean", 0), ("ball", 10), ("ball+player", 11))


def _arrange(n_total, lengths_fn, frac, rng):
    target = int(round(frac * n_total))
    lens = []
    while sum(lens) < target:
        lens.append(max(1, int(round(lengths_fn(rng)))))
    if not lens:
        return np.zeros(n_total, bool)
    lens[-1] -= sum(lens) - target
    lens = [x for x in lens if x > 0]
    cuts = np.sort(rng.integers(0, n_total - target + 1, size=len(lens)))
    spacing = np.diff(np.concatenate(([0], cuts)))
    mask = np.zeros(n_total, bool)
    pos = 0
    for s, g in zip(spacing, lens):
        pos += int(s)
        mask[pos:pos + g] = True
        pos += g
    return mask


def degrade(trk, blocks, condition, seed, match_idx):
    """Copy of trk with simulated ball (and player) dropout. Deterministic in (seed, match_idx)."""
    trk = trk.copy()
    if condition == "clean":
        return trk
    in_period = np.zeros(len(trk), bool)
    for _, a, b in blocks:
        in_period[a:b + 1] = True
    idx = np.flatnonzero(in_period)
    rng = np.random.default_rng([seed, match_idx, 0])
    gap = _arrange(len(idx), lambda r: r.lognormal(np.log(GAP_MEDIAN), GAP_SIGMA), BALL_MISSING, rng)
    trk[idx[gap], -1, :] = np.nan
    if condition == "ball+player":
        rng = np.random.default_rng([seed, match_idx, 1])
        for k in range(trk.shape[1] - 1):
            on = np.flatnonzero(np.isfinite(trk[:, k, 0]) & in_period)
            if len(on):
                m = _arrange(len(on), lambda r: r.lognormal(np.log(PL_MEDIAN), PL_SIGMA), PL_MISSING, rng)
                trk[on[m], k, :] = np.nan
    return trk


def match_rows(path, match_idx, step):
    z = np.load(path)
    blocks = [tuple(int(v) for v in b) for b in z["blocks"]]
    Xs, ys = [], []
    for cond, seed in TRAIN_VERSIONS:
        X, y = training_rows(degrade(z["trk"], blocks, cond, seed, match_idx), z["team"], z["gt"], step)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--cache", action="append", required=True, help="dir of <match>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()
    Xs, ys = [], []
    idx = 0
    for cache in args.cache:
        for path in sorted(glob.glob(os.path.join(cache, "*.npz"))):
            X, y = match_rows(path, idx, args.step)
            idx += 1
            Xs.append(X)
            ys.append(y)
            print(os.path.basename(path), len(y), flush=True)
    model = make_model().fit(np.concatenate(Xs), np.concatenate(ys))
    with open(args.out, "wb") as handle:
        pickle.dump(model, handle)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
