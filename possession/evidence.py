"""Per-frame evidence scores for the three states H, A, D.

Scores are additive log-scale votes: S[t, k] is added to the path score whenever the decoded state
at frame t is k. Only differences between states matter. Each component is documented with the
football rule it encodes. All thresholds live in params.PARAMS.
"""

import numpy as np

from .data import AWAY, FPS, HOME, PITCH, TEAM_OF_TAG, run_filter

H, A, D = 0, 1, 2

OFFENSIVE_CODES = {"build up", "progression", "creating chances", "ball in final third",
                   "ball in the box", "players in the box", "long ball", "attacking transition"}
DEFENSIVE_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}


def _centred_mean(values, window):
    """Centred moving mean of a 1-D array ignoring NaN, window in frames (odd-ish)."""
    finite = np.isfinite(values)
    v = np.where(finite, values, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(finite)))
    half = window // 2
    idx = np.arange(len(values))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(values), idx + half + 1)
    count = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)


def ball_speed(ball):
    """Ball speed (m/s) from a centred +-2 frame difference; NaN where either end is missing."""
    v = np.full(len(ball), np.nan)
    diff = ball[4:] - ball[:-4]
    v[2:-2] = np.hypot(diff[:, 0], diff[:, 1]) * FPS / 4.0
    return v


def geometry(players, speed, ball, team):
    """Team-wise nearest distances to the ball and the nearest player's speed."""
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    d_home = np.where(team[None, :] == HOME, d, np.inf).min(axis=1)
    d_away = np.where(team[None, :] == AWAY, d, np.inf).min(axis=1)
    nearest = np.argmin(d, axis=1)
    near_speed = speed[np.arange(len(d)), nearest]
    near_team = team[nearest]
    return d_home, d_away, near_team, near_speed


def control_evidence(players, speed, ball, source, team, p):
    """1) Ball control: who has the ball, as opposed to who is merely nearest.

    control   nearest player within R_CTRL, ball moving with them (slow, or at the player's speed),
              sustained >= MIN_CTRL frames for the same team  -> strong vote for that team
    contested both teams within R_CONT                         -> weak vote for the nearer team
    near      someone within R_NEAR, ball not in flight         -> very weak vote
    flight    ball faster than V_FLIGHT, nobody within R_CONT   -> no team vote (a pass in flight
              does not change possession); small vote for 'alive'
    still     ball nearly stationary for >= STILL_MIN frames    -> no team vote (set-piece setup)
    Frames from the auxiliary ball source are weighted by AUX_WEIGHT.
    """
    n = len(ball)
    S = np.zeros((n, 3))
    has = source > 0
    vb = ball_speed(ball)
    d_home, d_away, near_team, near_speed = geometry(players, speed, ball, team)
    d_min = np.minimum(d_home, d_away)

    still = run_filter(has & (vb < p["V_STILL"]), p["STILL_MIN"])
    slow = (vb <= p["V_CTRL_ABS"]) | (np.abs(vb - near_speed) <= p["V_CTRL_REL"])
    ctrl_raw = has & ~still & (d_min <= p["R_CTRL"]) & np.nan_to_num(slow, nan=0).astype(bool)
    ctrl = np.zeros(n, bool)
    for k in (HOME, AWAY):
        ctrl |= run_filter(ctrl_raw & (near_team == k), p["MIN_CTRL"])
    contested = has & ~ctrl & ~still & (d_home <= p["R_CONT"]) & (d_away <= p["R_CONT"])
    flight = has & ~still & (vb > p["V_FLIGHT"]) & (d_min > p["R_CONT"])
    near = has & ~ctrl & ~contested & ~still & ~flight & (d_min <= p["R_NEAR"])

    weight = np.where(source == 2, p["AUX_WEIGHT"], 1.0)
    rows = np.arange(n)
    for mask, w in ((ctrl, p["W_CTRL"]), (contested, p["W_CONT"]), (near, p["W_NEAR"])):
        idx = rows[mask]
        S[idx, near_team[mask]] += w * weight[mask]
    S[flight, D] -= p["W_FLIGHT_ALIVE"] * weight[flight]

    info = {"ctrl": ctrl, "contested": contested, "flight": flight, "near": near, "still": still,
            "near_team": near_team}
    return S, info


def deadball_evidence(players, speed, ball, source, team, p, still):
    """2) Dead ball: the ball is out, the play has stopped, or the teams line up for a kickoff.

    out       ball (Metrica only) beyond the lines by > M_OUT metres            -> strong D
    still     ball stationary for >= STILL_MIN frames                          -> D
    motion    mean player speed over a MOTION_WIN window below/above V_MID      -> +/- D
    kickoff   every team on its own side of the halfway line                  -> D
    """
    n = len(ball)
    S = np.zeros((n, 3))
    L, W = PITCH
    m = p["M_OUT"]
    metrica = source == 1
    bx, by = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = metrica & ((bx < -m) | (bx > L + m) | (by < -m) | (by > W + m))
    S[out, D] += p["W_OUT"]
    S[still, D] += p["W_STILL"]

    mean_speed = _row_nanmean(speed)
    smooth = _centred_mean(mean_speed, p["MOTION_WIN"])
    vote = np.clip((p["V_MID"] - smooth) / p["V_SCALE"], -1.0, 1.0)
    S[:, D] += p["W_MOTION"] * np.nan_to_num(vote, nan=0.0)

    # Kickoff formation: home and away fully separated by the halfway line (either orientation).
    x = players[:, :, 0]
    present = np.isfinite(x)
    half = L / 2.0
    left = present & (x < half - p["KO_MARGIN"])
    right = present & (x > half + p["KO_MARGIN"])
    counts = {}
    for k in (HOME, AWAY):
        sel = team[None, :] == k
        counts[k] = (np.sum(present & sel, axis=1), np.sum(left & sel, axis=1),
                     np.sum(right & sel, axis=1))
    enough = (counts[HOME][0] >= p["KO_MIN_PLAYERS"]) & (counts[AWAY][0] >= p["KO_MIN_PLAYERS"])
    tol = p["KO_TOL"]  # players allowed on the wrong side (the kicker stands on the line)
    split_a = (counts[HOME][1] >= counts[HOME][0] - tol) & (counts[AWAY][2] >= counts[AWAY][0] - tol)
    split_b = (counts[HOME][2] >= counts[HOME][0] - tol) & (counts[AWAY][1] >= counts[AWAY][0] - tol)
    kickoff = run_filter(enough & (split_a | split_b), p["KO_MIN_LEN"])
    S[kickoff, D] += p["W_KO"]
    return S, {"out": out, "kickoff": kickoff, "motion": smooth}


def _row_nanmean(a):
    finite = np.isfinite(a)
    count = finite.sum(axis=1)
    total = np.where(finite, a, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def setpiece_evidence(n, tags, p):
    """3) Set pieces: dead before the restart, then the tagged team restarts.

    The tag window opens about SP_OFFSET frames before the restart (measured on the tagged
    kickoffs and on the player-speed profile around every set-piece tag; no labels involved).
    Throw-ins have Team = N/A, so they contribute the dead-ball part only.
    """
    S = np.zeros((n, 3))
    for e in tags:
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        restart = e["start_frame"] + p["SP_OFFSET"]
        a = max(0, e["start_frame"] - p["SP_PRE"])
        S[a:max(a, restart), D] += p["W_SP_DEAD"]
        b = min(n, restart + p["SP_POST"])
        S[restart:b, D] -= p["W_SP_ALIVE"]
        side = TEAM_OF_TAG.get((e["tags"].get("Team") or "").strip().lower())
        if side is not None:
            S[restart:b, side] += p["W_SP_TEAM"]
    return S


def period_start_evidence(n, tags, blocks, p):
    """5) Every half starts with a kickoff: graded frames of a half before its tagged
    'Kick Off (Start)' restart are dead ball (players walking out, warm-up balls).

    A football rule, applied through the automatic tag of that half -- no frame is hard-coded.
    """
    S = np.zeros((n, 3))
    for _, first, last in blocks:
        restarts = [e["start_frame"] + p["SP_OFFSET"] for e in tags
                    if (e["code"] or "").strip().lower() == "set pieces"
                    and (e["tags"].get("Type") or "").strip().lower() == "kick off (start)"
                    and first - p["SP_OFFSET"] <= e["start_frame"] <= last]
        if restarts:
            restart = min(restarts)
            if restart > first:
                S[first:restart, D] += p["W_PRESTART"]
    return S


def tag_votes(n, tags):
    """{code: (n,) array of +1 home / -1 away / 0} -- what each phase code says about possession."""
    out = {}
    for e in tags:
        code = (e["code"] or "").strip().lower()
        side = TEAM_OF_TAG.get((e["tags"].get("Team") or "").strip().lower())
        if side is None or code not in OFFENSIVE_CODES | DEFENSIVE_CODES:
            continue
        vote = 1.0 if side == HOME else -1.0
        if code in DEFENSIVE_CODES:
            vote = -vote
        out.setdefault(code, np.zeros(n))[e["start_frame"]:e["end_frame"]] += vote
    return out


def reliability_weights(votes, proxy_team, proxy_mask, ref_acc, min_acc=0.0):
    """Label-free self-calibration of each evidence source.

    proxy_team (+1/-1) on proxy_mask frames is our own sustained-control call where the Metrica
    ball is visible. A source agreeing with it at rate acc gets weight logit(acc) / logit(ref_acc)
    (0 if acc <= 0.5). No human label is involved; the frames are chosen by the algorithm.
    """
    weights = {}
    ref = np.log(ref_acc / (1 - ref_acc))
    for code, v in votes.items():
        v = np.sign(v)
        m = proxy_mask & (v != 0)
        if m.sum() < 100:
            weights[code] = 1.0
            continue
        acc = float(np.mean(v[m] == proxy_team[m]))
        acc = min(max(acc, 1e-3), 1 - 1e-3)
        weights[code] = 0.0 if acc < min_acc else max(0.0, np.log(acc / (1 - acc))) / ref
    return weights


def tag_evidence(n, tags, p, weights=None):
    """4) Smart Tagging phases as a weak team prior (strongest where the ball is missing).

    Offensive phases vote for the tagged team; defensive phases vote for the opponent. With
    weights (see reliability_weights), each code's vote is scaled by its measured reliability.
    """
    net = np.zeros(n)  # + home, - away
    for code, v in tag_votes(n, tags).items():
        net += v * (1.0 if weights is None else weights[code])
    S = np.zeros((n, 3))
    net = np.clip(net, -p["TAG_CLIP"], p["TAG_CLIP"])
    S[:, H] += p["W_TAG"] * np.maximum(net, 0)
    S[:, A] += p["W_TAG"] * np.maximum(-net, 0)
    return S


def pathcrf_evidence(p_home, source, p, weight):
    """6) PathCRF (pretrained, ball-free) team probability, used where the ball is missing.

    p_home = P(H | alive) per frame (NaN where unavailable). Tempered towards 1/2 by PCRF_EPS
    because the raw output is overconfident; the more likely team gets 0, the other team gets
    W_PCRF * weight * log(p_other / p_best) -- a pure H-vs-A vote that never favours D.
    """
    n = len(source)
    S = np.zeros((n, 3))
    m = np.isfinite(p_home) & (source == 0)
    q = (1 - p["PCRF_EPS"]) * p_home[m] + p["PCRF_EPS"] / 2
    lh, la = np.log(q), np.log(1 - q)
    best = np.maximum(lh, la)
    S[m, H] += p["W_PCRF"] * weight * (lh - best)
    S[m, A] += p["W_PCRF"] * weight * (la - best)
    return S
