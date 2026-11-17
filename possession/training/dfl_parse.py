"""Parse one Sportec Open DFL match into the possession pipeline's input contract.

usage: python -I parse_match.py META_XML RAW_XML OUT_NPZ

Writes:
  trk     float32 (n, n_slots+1, 3): x, y normalised 0-1, speed m/s; ball last; NaN = missing
  team    int8 (n_slots,): 0 home, 1 away
  gt      <U1 (n,): H/A/D on period frames, X on the gap frames between periods
  blocks  int64 (n_periods, 3): period, first, last (inclusive)
  plus raw diagnostics (raw BallPossession / BallStatus strings from a direct lxml pass).
"""

import collections
import json
import sys
import time

import numpy as np
from lxml import etree

from kloppy import sportec
from kloppy.domain import BallState

GAP = 250  # X frames inserted between periods so the metric sees separate runs


def raw_ball_attrs(raw_path):
    """Direct pass over the BALL framesets: raw BallPossession / BallStatus strings per frame."""
    out = {}
    ctx = etree.iterparse(raw_path, events=("start", "end"), huge_tree=True)
    cur = None
    for ev, el in ctx:
        if ev == "start" and el.tag == "FrameSet":
            cur = el.get("GameSection") if el.get("TeamId") == "BALL" else None
        elif ev == "start" and el.tag == "Frame" and cur is not None:
            out[(cur, int(el.get("N")))] = (el.get("BallPossession"), el.get("BallStatus"))
        elif ev == "end" and el.tag in ("Frame", "FrameSet"):
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    return out


def main(meta_path, raw_path, out_path):
    t0 = time.time()
    ds = sportec.load_tracking(raw_data=raw_path, meta_data=meta_path,
                               coordinates="sportec", only_alive=False)
    t_load = time.time() - t0
    md = ds.metadata
    L = md.pitch_dimensions.pitch_length
    W = md.pitch_dimensions.pitch_width
    home, away = md.teams
    players = [pl for t in (home, away) for pl in t.players]
    slot = {pl.player_id: i for i, pl in enumerate(players)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    n_slots = len(players)

    # Frame ids per period.
    by_period = collections.defaultdict(list)
    for fr in ds.records:
        by_period[fr.period.id].append(fr)
    periods = sorted(by_period)
    spans, first = [], 0
    for pid in periods:
        fids = [fr.frame_id for fr in by_period[pid]]
        lo, hi = min(fids), max(fids)
        spans.append((pid, lo, hi, first))
        first += (hi - lo + 1) + GAP
    n = first - GAP

    trk = np.full((n, n_slots + 1, 3), np.nan, dtype=np.float32)
    s_raw = np.full((n, n_slots), np.nan, dtype=np.float32)
    gt = np.full(n, "X", dtype="<U1")
    owner = np.full(n, -1, dtype=np.int8)
    alive = np.full(n, -1, dtype=np.int8)
    in_period = np.zeros(n, bool)
    frame_id = np.full(n, -1, dtype=np.int64)
    period_of = np.zeros(n, dtype=np.int8)
    blocks = []
    for pid, lo, hi, off in spans:
        blocks.append((pid, off, off + hi - lo))
        in_period[off:off + hi - lo + 1] = True
        frame_id[off:off + hi - lo + 1] = np.arange(lo, hi + 1)
        period_of[off:off + hi - lo + 1] = pid
        for fr in by_period[pid]:
            i = off + fr.frame_id - lo
            b = fr.ball_coordinates
            trk[i, -1, 0] = (b.x + L / 2) / L
            trk[i, -1, 1] = (b.y + W / 2) / W
            for pl, pd in fr.players_data.items():
                if pd.coordinates is None:
                    continue
                k = slot[pl.player_id]
                trk[i, k, 0] = (pd.coordinates.x + L / 2) / L
                trk[i, k, 1] = (pd.coordinates.y + W / 2) / W
                if pd.speed is not None:
                    s_raw[i, k] = pd.speed
            o = 0 if fr.ball_owning_team == home else (1 if fr.ball_owning_team == away else -1)
            a = 1 if fr.ball_state == BallState.ALIVE else 0
            owner[i], alive[i] = o, a
            gt[i] = "D" if a == 0 else ("H" if o == 0 else "A")
    missing_frames = int(np.sum(in_period & (alive < 0)))
    # Frames inside a period with no record: keep X in GT (should not happen).

    # Speed: finite difference of positions (m/s), centred over 5 frames, vs the S attribute.
    xy_m = trk[:, :-1, :2].astype(np.float64) * np.array([L, W])
    v = np.full((n, n_slots), np.nan)
    d = xy_m[4:] - xy_m[:-4]
    v[2:-2] = np.hypot(d[..., 0], d[..., 1]) * 25.0 / 4.0
    for _, a0, b0 in blocks:  # no differences across the period gap
        v[a0:a0 + 2] = np.nan
        v[b0 - 1:b0 + 1] = np.nan
    ok = np.isfinite(v) & np.isfinite(s_raw) & (v > 1.0)
    ratio = float(np.median(s_raw[ok] / v[ok])) if ok.any() else float("nan")
    # Use S converted to m/s when it is km/h (ratio ~3.6), otherwise the finite difference.
    if abs(ratio - 3.6) < 0.3:
        speed = s_raw / 3.6
        speed_src = "S/3.6"
    elif abs(ratio - 1.0) < 0.1:
        speed = s_raw
        speed_src = "S"
    else:
        speed = v.astype(np.float32)
        speed_src = "finite-diff"
    trk[:, :-1, 2] = speed
    trk[:, -1, 2] = np.nan

    # Raw attribute check.
    t1 = time.time()
    raw = raw_ball_attrs(raw_path)
    sec_name = {1: "firstHalf", 2: "secondHalf", 3: "firstHalfExtra", 4: "secondHalfExtra"}
    poss_vals = collections.Counter(v[0] for v in raw.values())
    stat_vals = collections.Counter(v[1] for v in raw.values())
    raw_poss = np.full(n, "", dtype="<U2")
    raw_stat = np.full(n, "", dtype="<U2")
    for pid, lo, hi, off in spans:
        for fid in range(lo, hi + 1):
            r = raw.get((sec_name[pid], fid))
            if r is not None:
                raw_poss[off + fid - lo] = r[0] or "-"
                raw_stat[off + fid - lo] = r[1] or "-"
    t_raw = time.time() - t1

    np.savez_compressed(out_path, trk=trk, team=team, gt=gt, blocks=np.array(blocks),
                        owner=owner, alive=alive, frame_id=frame_id, period=period_of,
                        raw_poss=raw_poss, raw_stat=raw_stat, speed_fd=v.astype(np.float32))
    info = {
        "match": md.game_id, "home": home.name, "away": away.name,
        "pitch": [L, W], "fps": md.frame_rate, "n_slots": n_slots, "n": n,
        "periods": [(pid, lo, hi, hi - lo + 1) for pid, lo, hi, _ in spans],
        "n_records": len(ds.records), "missing_in_period": missing_frames,
        "speed_ratio_S_over_fd": ratio, "speed_src": speed_src,
        "raw_BallPossession": dict(poss_vals), "raw_BallStatus": dict(stat_vals),
        "players_present_mean": [float(np.mean(np.sum(np.isfinite(trk[in_period][:, :-1, 0][:, team == k]),
                                                       axis=1))) for k in (0, 1)],
        "ball_x_range": [float(np.nanmin(trk[:, -1, 0])), float(np.nanmax(trk[:, -1, 0]))],
        "ball_y_range": [float(np.nanmin(trk[:, -1, 1])), float(np.nanmax(trk[:, -1, 1]))],
        "t_kloppy_s": round(t_load, 1), "t_rawpass_s": round(t_raw, 1),
        "t_total_s": round(time.time() - t0, 1),
    }
    with open(out_path.replace(".npz", ".json"), "w") as h:
        json.dump(info, h, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main(*sys.argv[1:4])
