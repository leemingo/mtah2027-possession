"""PathCRF (Kim et al., KDD 2026) as a ball-free team-possession prior: P(home | alive) per frame.

    bash possession/setup_pathcrf.sh                       # clone PathCRF at the pinned commit
    pip install -r requirements-pathcrf.txt                # torch CPU + torch_geometric (Python 3.11)
    python -m possession.pathcrf_adapter --repo external/pathcrf --out data/cache/pathcrf_p_home.npy

Uses the authors' pretrained checkpoint saved/140 (dynamic sparse CRF, trained on the Sportec Open
DFL data) unmodified; nothing is trained here. Steps per graded half:

1. Tracklets -> 11 pseudo-players per team: merge duplicate detections (< 0.8 m), goalkeeper = the
   deepest detection within 35 m of its own goal line, the other 10 slots by per-frame Hungarian
   assignment to each slot's last position, gaps filled by linear interpolation.
2. Kinematics exactly as PathCRF's calculate_running_features (savgol velocity / acceleration,
   > 12 m/s jumps removed), coordinates on PathCRF's 105 x 68 m pitch, left team first.
3. Model emissions on 10 s windows at 5 fps, blended as in PathCRF's inference_episode.
4. CRF forward-backward over the whole half (our own implementation; PathCRF's Viterbi/logZ use an
   absolute forbid score that breaks on sequences longer than ~6 minutes), giving edge marginals.
5. P(team) = mass of edges whose source node belongs to that team (last toucher's team);
   P(home | alive) = P_home / (P_home + P_away), interpolated back to 25 fps.

Deterministic on CPU (identical across runs and thread counts). The output is frame-level derived
data and lives in data/cache/ (gitignored).
"""

import argparse
import hashlib
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from . import data

warnings.filterwarnings("ignore")

PX, PY = 105.0, 68.0          # PathCRF's pitch (datatools/config.py)
FPS, SAMPLE_FREQ, TEAM_SIZE = 25, 5, 11
CHECKPOINT = os.path.join("saved", "140")
WINDOW_SECONDS = 10.0


def dedupe(p, thr=0.8):
    """Merge detections closer than thr metres (duplicate tracklets of one player)."""
    if len(p) < 2:
        return p
    used = np.zeros(len(p), bool)
    out = []
    for i in range(len(p)):
        if used[i]:
            continue
        grp = (np.linalg.norm(p - p[i], axis=1) < thr) & ~used
        used |= grp
        out.append(p[grp].mean(0))
    return np.array(out)


def pseudo_players(xy, team_mask, left_side):
    """(T, 11, 2) positions, slot 0 = goalkeeper, gaps interpolated."""
    T = len(xy)
    pos = np.full((T, TEAM_SIZE, 2), np.nan)
    last = np.full((TEAM_SIZE, 2), np.nan)
    for t in range(T):
        p = xy[t][team_mask]
        p = dedupe(p[np.isfinite(p).all(1)])
        if len(p) == 0:
            continue
        g = np.argmin(p[:, 0]) if left_side else np.argmax(p[:, 0])
        depth = p[g, 0] if left_side else PX - p[g, 0]
        if depth < 35.0:
            pos[t, 0] = last[0] = p[g]
            p = np.delete(p, g, axis=0)
        if len(p) == 0:
            continue
        init = np.isfinite(last[1:, 0])
        cost = np.where(init[None, :],
                        np.linalg.norm(p[:, None, :] - last[None, 1:, :], axis=2), 8.0)
        r, c = linear_sum_assignment(cost)
        pos[t, 1 + c] = p[r]
        last[1 + c] = p[r]
    for k in range(TEAM_SIZE):
        for d in range(2):
            s = pd.Series(pos[:, k, d])
            if s.notna().any():
                pos[:, k, d] = s.interpolate(limit_direction="both").to_numpy()
    centroid = np.nanmean(pos, axis=1, keepdims=True)
    return np.where(np.isfinite(pos), pos, np.broadcast_to(centroid, pos.shape))


def running_features(x, y, thr=12.0):
    """PathCRF datatools.utils.calculate_running_features(denoise=True), same maths."""
    x, y = pd.Series(x.copy()), pd.Series(y.copy())
    out = np.sqrt(x.diff() ** 2 + y.diff() ** 2) * FPS > thr
    if out.any():
        x[out] = np.nan
        y[out] = np.nan
        x = x.interpolate().bfill().ffill()
        y = y.interpolate().bfill().ffill()
    x, y = x.to_numpy(), y.to_numpy()
    vx = savgol_filter(np.diff(x) * FPS, window_length=15, polyorder=2)
    vy = savgol_filter(np.diff(y) * FPS, window_length=15, polyorder=2)
    ax = savgol_filter(np.diff(vx) * FPS, window_length=9, polyorder=2)
    ay = savgol_filter(np.diff(vy) * FPS, window_length=9, polyorder=2)
    VX, VY, AC = np.empty(len(x)), np.empty(len(x)), np.zeros(len(x))
    VX[1:], VY[1:] = vx, vy
    VX[0], VY[0] = vx[0], vy[0]
    AC[1:-1] = np.sqrt(ax ** 2 + ay ** 2)
    return x, y, VX, VY, np.sqrt(VX ** 2 + VY ** 2), AC


def load_model(repo):
    import torch
    from models.utils import build_model, load_trial_args, resolve_model_path
    path = os.path.join(repo, CHECKPOINT)
    model = build_model(load_trial_args(path), device="cpu")
    model.load_state_dict(torch.load(resolve_model_path(path, "state_dict_best_acc.pt"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    return model


def half_p_home(model, xy, is_home, home_left):
    """P(home | alive) for every frame of one half. xy: (T, 46, 2) metres on 105 x 68."""
    import torch
    import inference as pinf

    T = len(xy)
    cols = {}
    for name, mask in (("home", is_home), ("away", ~is_home)):
        left_side = home_left if name == "home" else not home_left
        pos = pseudo_players(xy, mask, left_side)
        for k in range(TEAM_SIZE):
            for f, v in zip(("x", "y", "vx", "vy", "speed", "accel"),
                            running_features(pos[:, k, 0], pos[:, k, 1])):
                cols["{}_{}_{}".format(name, k, f)] = v
    tracking = pd.DataFrame(cols, index=pd.RangeIndex(0, T, name="frame_id"))
    left, right = ("home", "away") if home_left else ("away", "home")

    ep = pinf._add_outside_nodes(tracking)
    order = (["{}_{}".format(left, k) for k in range(TEAM_SIZE)]
             + ["{}_{}".format(right, k) for k in range(TEAM_SIZE)]
             + ["out_left", "out_right", "out_bottom", "out_top"])
    order = sorted(order[:TEAM_SIZE]) + sorted(order[TEAM_SIZE:2 * TEAM_SIZE]) + order[2 * TEAM_SIZE:]
    feats = [p + f for p in order for f in ("_x", "_y", "_vx", "_vy", "_speed", "_accel")]
    assert not ep[feats].isna().any().any()
    ind = pinf._build_node_indicators(order, left, right, left + "_0", right + "_0")
    X = np.concatenate([np.broadcast_to(ind[None], (T, len(order), 2)),
                        ep[feats].to_numpy(np.float32).reshape(T, len(order), -1)], -1)
    down = np.arange(0, T, SAMPLE_FREQ)
    if down[-1] != T - 1:
        down = np.append(down, T - 1)
    Xd, _, padded = pinf._pad_nodes(X[down], order, left, right, TEAM_SIZE)
    assert padded == order
    Td = len(Xd)

    wl = int(round(WINDOW_SECONDS * FPS / SAMPLE_FREQ))
    stride = int(round(0.5 * WINDOW_SECONDS * FPS / SAMPLE_FREQ))
    if Td <= wl:
        windows = [(0, Td)]
    else:
        starts = list(range(0, Td - wl + 1, stride))
        windows = [(s, Td if i == len(starts) - 1 else s + wl) for i, s in enumerate(starts)]

    valid = model.valid_orig_edge_ids
    K = valid.numel()
    micro = torch.zeros(Td, 676)
    emb = torch.zeros(Td, K, 16)
    wsum = torch.zeros(Td)
    with torch.no_grad():
        for i, (s, e) in enumerate(windows):
            _, mic, em = pinf._run_model(model, Xd[s:e], "cpu")
            w = pinf._window_weights(windows, i)
            micro[s:e] += mic * w[:, None]
            emb[s:e] += em[0].index_select(1, valid) * w[:, None, None]
            wsum[s:e] += w
        den = torch.where(wsum == 0, torch.ones_like(wsum), wsum)
        emis = (micro / den[:, None]).index_select(1, valid).double()
        emb = emb / den[:, None, None]

        # CRF forward-backward. The transition score linear(cat(e_prev, e_curr)) is separable.
        Wt = model.crf.linear.weight.detach()[0]
        bias = float(model.crf.linear.bias.detach()[0])
        a_prev = (emb @ Wt[:16]).double()
        c_curr = (emb @ Wt[16:]).double()
        inc_idx, inc_mask = model.crf.inc_idx, model.crf.inc_mask
        allowed = torch.zeros(K, K, dtype=torch.bool)
        jj = torch.arange(K)[:, None].expand_as(inc_idx)
        allowed[inc_idx[inc_mask], jj[inc_mask]] = True
        neg = torch.where(allowed, 0.0, -1e4).double()
        alpha = torch.empty(Td, K, dtype=torch.float64)
        beta = torch.zeros(Td, K, dtype=torch.float64)
        alpha[0] = emis[0]
        for t in range(1, Td):
            alpha[t] = emis[t] + c_curr[t] + bias + torch.logsumexp(
                (alpha[t - 1] + a_prev[t - 1])[:, None] + neg, dim=0)
        for t in range(Td - 1, 0, -1):
            beta[t - 1] = a_prev[t - 1] + bias + torch.logsumexp(
                neg + (emis[t] + c_curr[t] + beta[t])[None, :], dim=1)
        post = torch.exp(alpha + beta - torch.logsumexp(alpha[-1], 0)).numpy()

    src = model.comp_src.numpy()
    p_left = post[:, src < TEAM_SIZE].sum(1)
    p_right = post[:, (src >= TEAM_SIZE) & (src < 2 * TEAM_SIZE)].sum(1)
    p_left = np.interp(np.arange(T), down, p_left)
    p_right = np.interp(np.arange(T), down, p_right)
    p_home, p_away = (p_left, p_right) if home_left else (p_right, p_left)
    return p_home / np.maximum(p_home + p_away, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.path.join(data.ROOT, "external", "pathcrf"))
    ap.add_argument("--out", default=os.path.join(data.DATA_DIR, "cache", "pathcrf_p_home.npy"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    sys.path.insert(0, os.path.abspath(args.repo))

    trk = data.load_tracking()
    is_home = data.slot_teams() == data.HOME
    model = load_model(os.path.abspath(args.repo))
    out = np.full(len(trk), np.nan)
    for half, first, last in data.graded_blocks():
        t0 = time.time()
        xy = trk[first:last + 1, :-1, :2].astype(np.float64) * np.array([PX, PY])
        # Orientation from the data: the side whose players stand further left over the half.
        home_left = bool(np.nanmean(xy[:, is_home, 0]) < np.nanmean(xy[:, ~is_home, 0]))
        out[first:last + 1] = half_p_home(model, xy, is_home, home_left)
        print("half {}: home {} , {:.0f} s".format(half, "left" if home_left else "right",
                                                   time.time() - t0), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, out)
    with open(args.out, "rb") as handle:
        print("wrote {}  md5 {}".format(args.out, hashlib.md5(handle.read()).hexdigest()))


if __name__ == "__main__":
    main()
