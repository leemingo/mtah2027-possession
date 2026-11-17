"""Generate a possession submission.

    python -m possession.run --out evaluation/results/A.csv                    # Metrica ball only
    python -m possession.run --aux ball_results.npz --aux-key wasb_full \
        --out evaluation/results/B.csv                                         # + video ball
    python -m possession.run --disable tags,setpiece --set C_SWITCH=20 ...     # ablations

Prints label-free diagnostics of the output. The public 5-minute ground truth is only scored when
--check-public is given, as a pipeline sanity check (never for choosing parameters).
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from . import data, decode, evidence
from .params import COMPONENTS, PARAMS


def trk_with_ball(trk, aux):
    """Copy of trk whose ball channel is the chosen ball source (Metrica, filled by aux)."""
    if aux is None:
        return trk
    ball, source = data.ball_source(trk, aux)
    out = trk.copy()
    fill = source == 2
    out[fill, -1, 0] = ball[fill, 0] / data.PITCH[0]
    out[fill, -1, 1] = ball[fill, 1] / data.PITCH[1]
    return out


def build_scores(trk, team, tags, p, enabled, aux=None, p_home=None, model=None):
    players, speed, _ = data.players_and_ball(trk)
    ball, source = data.ball_source(trk, aux)
    n = len(trk)
    S = np.zeros((n, 3))
    if model is not None:
        from . import learned
        P = learned.predict_proba(model, learned.features(trk_with_ball(trk, aux), team))
        S += learned.learned_evidence(P, p)
    S_ctrl, info = evidence.control_evidence(players, speed, ball, source, team, p)
    # Label-free calibration frames: sustained control on the Metrica ball.
    proxy_mask = info["ctrl"] & (source == 1)
    proxy_team = np.where(info["near_team"] == data.HOME, 1.0, -1.0)
    info["weights"] = {}
    if "control" in enabled:
        S += S_ctrl
    if "deadball" in enabled:
        S_dead, dinfo = evidence.deadball_evidence(players, speed, ball, source, team, p,
                                                   info["still"])
        S += S_dead
        info.update(dinfo)
    if "setpiece" in enabled:
        S += evidence.setpiece_evidence(n, tags, p)
        S += evidence.period_start_evidence(n, tags, data.graded_blocks(), p)
    if "tags" in enabled:
        weights = None
        if p["TAG_RELIABILITY"]:
            weights = evidence.reliability_weights(evidence.tag_votes(n, tags), proxy_team,
                                                   proxy_mask, p["REL_REF_ACC"], p["TAG_REL_MIN"])
            info["weights"].update({k: round(v, 3) for k, v in weights.items()})
        S += evidence.tag_evidence(n, tags, p, weights)
    if p_home is not None:
        vote = np.where(np.isfinite(p_home), np.where(p_home >= 0.5, 1.0, -1.0), 0.0)
        w = evidence.reliability_weights({"pathcrf": vote}, proxy_team, proxy_mask,
                                         p["REL_REF_ACC"])["pathcrf"]
        info["weights"]["pathcrf"] = round(w, 3)
        S += evidence.pathcrf_evidence(p_home, source, p, w)
    info["source"] = source
    return S, info


def spells(labels, blocks):
    out = []
    for _, a, b in blocks:
        x = labels[a:b + 1]
        br = np.flatnonzero(x[1:] != x[:-1]) + 1
        bd = np.concatenate(([0], br, [len(x)]))
        out += [(a + bd[i], a + bd[i + 1], x[bd[i]]) for i in range(len(bd) - 1)]
    return out


def diagnostics(labels, blocks, info):
    graded = np.zeros(len(labels), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    segs = spells(labels, blocks)
    poss = [(s, e, k) for s, e, k in segs if k in "HA"]
    dur = np.array([e - s for s, e, _ in poss]) / data.FPS
    lab = labels[graded]
    alive = np.isin(lab, ["H", "A"])
    src = info["source"][graded]
    # Label-free consistency: on frames with sustained ball control, does the output name the
    # controlling team? (Low = too sticky; it is not a ground truth.)
    ctrl = info["ctrl"] & graded
    team_letter = np.where(info["near_team"] == data.HOME, "H", "A")
    ctrl_agree = float((labels[ctrl] == team_letter[ctrl]).mean()) if ctrl.any() else 0.0
    n_switch, n_switch_gap = 0, 0
    for (s1, e1, k1), (s2, e2, k2) in zip(segs[:-1], segs[1:]):
        if e1 == s2 and {k1, k2} == {"H", "A"}:
            n_switch += 1
            n_switch_gap += int(info["source"][s2] == 0)
    return {
        "ctrl_agree": round(ctrl_agree, 3),
        "team_switches": n_switch,
        "switch_in_gap": n_switch_gap,
        "possessions": len(poss),
        "dur_median_s": round(float(np.median(dur)), 2) if len(dur) else 0,
        "dur_mean_s": round(float(dur.mean()), 2) if len(dur) else 0,
        "share_lt2s": round(float((dur < 2).mean()), 3) if len(dur) else 0,
        "dead_segments": sum(1 for _, _, k in segs if k == "D"),
        "D_share": round(float((lab == "D").mean()), 3),
        "home_share_alive": round(float((lab == "H")[alive].mean()), 3) if alive.any() else 0,
        "ball_metrica": round(float((src == 1).mean()), 3),
        "ball_aux": round(float((src == 2).mean()), 3),
    }


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def check_public(labels):
    sys.path.insert(0, os.path.join(data.ROOT, "evaluation"))
    import possession_metrics as pm
    gt = pm.label_codes(pm.read_label_csv(
        os.path.join(data.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    r = pm.evaluate(gt, pm.label_codes(labels))
    return {"public_spellF1": round(r["spell_f1_mean"], 4), "public_macroF1": round(r["macro_f1"], 4),
            "public_poss": r["n_pred_spells"], "public_gt_poss": r["n_gt_spells"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--aux", help="ball-detection bundle (.npz) for the auxiliary ball")
    ap.add_argument("--aux-key", default="wasb_full")
    ap.add_argument("--pathcrf", help=".npy of per-frame P(home | alive) from the PathCRF adapter")
    ap.add_argument("--model", help="pickled frame classifier from possession.learned (other matches)")
    ap.add_argument("--disable", default="", help="comma list of " + ",".join(COMPONENTS))
    ap.add_argument("--set", action="append", default=[], help="PARAM=value override")
    ap.add_argument("--check-public", action="store_true",
                    help="score the public 5 minutes (pipeline sanity check only)")
    args = ap.parse_args(argv)

    p = dict(PARAMS)
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in p:
            raise SystemExit("unknown parameter " + key)
        p[key] = type(p[key])(float(value)) if isinstance(p[key], int) else float(value)
    disabled = {c for c in args.disable.split(",") if c}
    enabled = [c for c in COMPONENTS if c not in disabled]

    trk = data.load_tracking()
    team = data.slot_teams()
    tags = data.smart_tags()
    blocks = data.graded_blocks()
    aux = data.aux_ball(args.aux, args.aux_key, p["AUX_CONF"], int(p["AUX_MIN"])) if args.aux else None

    p_home = np.load(args.pathcrf) if args.pathcrf else None
    model = None
    if args.model:
        import pickle
        with open(args.model, "rb") as handle:
            model = pickle.load(handle)
    S, info = build_scores(trk, team, tags, p, enabled, aux, p_home, model)
    labels = decode.decode(S, blocks, decode.transition_costs(p), len(trk))
    md5 = write_csv(args.out, labels)

    report = {"out": args.out, "md5": md5, "enabled": enabled,
              "overrides": args.set, "aux": [args.aux, args.aux_key] if args.aux else None,
              "pathcrf": args.pathcrf, "model": args.model, "weights": info["weights"]}
    report.update(diagnostics(labels, blocks, info))
    if args.check_public:
        report.update(check_public(labels))
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
