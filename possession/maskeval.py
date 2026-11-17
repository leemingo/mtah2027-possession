"""Label-free model selection: mask the Metrica ball where it IS visible and see how well each
variant recovers what the unmasked ball says.

Synthetic gaps are drawn (fixed seeds) inside ball-visible play with lengths resampled from the
real gap-length distribution, covering MASK_SHARE of the ball-visible graded frames. Each variant
is run on the masked data; scored only on masked frames:

  team_acc   frames with sustained Metrica-ball control (unmasked): output team == controlling team
  alive_acc  masked frames: output alive/dead == the variant's own unmasked output
  self_spell spellF1 of the masked output against the variant's own unmasked output
             (official metric code, restricted to the graded halves)

No ground truth, public or private, is read. The pseudo-references come from the algorithm itself
on unmasked data, which is allowed for the provided game (RULES.md: no person decides any label).

    python -m possession.maskeval --variants A,Ar,P --pathcrf data/cache/pathcrf_p_home.npy
"""

import argparse
import json
import os
import sys

import numpy as np

from . import data, decode
from .params import COMPONENTS, PARAMS
from .run import build_scores

sys.path.insert(0, os.path.join(data.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (read-only use of the official metric code)

MASK_SHARE = 0.15
SEEDS = (0, 1, 2)


def gap_lengths(trk, blocks):
    seen = np.isfinite(trk[:, -1, 0])
    out = []
    for _, a, b in blocks:
        m = ~seen[a:b + 1]
        e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))))
        out += (e[1::2] - e[0::2]).tolist()
    return np.array([g for g in out if g >= 25])


def synthetic_mask(trk, blocks, seed):
    rng = np.random.RandomState(seed)
    seen = np.isfinite(trk[:, -1, 0])
    graded = np.zeros(len(trk), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    lengths = gap_lengths(trk, blocks)
    target = MASK_SHARE * (seen & graded).sum()
    mask = np.zeros(len(trk), bool)
    candidates = np.flatnonzero(seen & graded)
    while mask.sum() < target:
        start = candidates[rng.randint(len(candidates))]
        length = int(lengths[rng.randint(len(lengths))])
        stop = min(start + length, len(trk))
        if not graded[start:stop].all():
            continue
        mask[start:stop] = True
    return mask & seen


VARIANTS = {
    "A": ({}, False, None),
    "Ar": ({"TAG_RELIABILITY": 1}, False, None),
    "P0": ({}, True, None),
    "P": ({"TAG_RELIABILITY": 1}, True, None),
    "B07": ({"AUX_CONF": 0.7}, False, "aux"),
    "B05": ({"AUX_CONF": 0.5}, False, "aux"),
    "PB05": ({"TAG_RELIABILITY": 1, "AUX_CONF": 0.5}, True, "aux"),
    "B03": ({"AUX_CONF": 0.3}, False, "aux"),
    "P0B07": ({"AUX_CONF": 0.7}, True, "aux"),
    "P0B05": ({"AUX_CONF": 0.5}, True, "aux"),
    "P0B03": ({"AUX_CONF": 0.3}, True, "aux"),
}


def labels_for(trk, team, tags, blocks, p, p_home, aux, model=None, enabled=None):
    enabled = list(COMPONENTS) if enabled is None else enabled
    S, info = build_scores(trk, team, tags, p, enabled, aux, p_home, model)
    return decode.decode(S, blocks, decode.transition_costs(p), len(trk)), info


def parse_variant(name):
    """'P0B05' -> rules only; 'L:P0B05' -> learned emission replaces control/dead-ball rules;
    'L+:P0B05' -> learned emission added to the rules."""
    if name.startswith("L+:"):
        return VARIANTS[name[3:]], "add"
    if name.startswith("L:"):
        return VARIANTS[name[2:]], "replace"
    return VARIANTS[name], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,Ar,P0,P")
    ap.add_argument("--pathcrf")
    ap.add_argument("--aux")
    ap.add_argument("--aux-key", default="wasb_full")
    ap.add_argument("--set", action="append", default=[], help="PARAM=value applied to all variants")
    ap.add_argument("--model", help="pickled classifier for L:/L+: variants")
    args = ap.parse_args()
    model = None
    if args.model:
        import pickle
        with open(args.model, "rb") as handle:
            model = pickle.load(handle)

    trk = data.load_tracking()
    team = data.slot_teams()
    tags = data.smart_tags()
    blocks = data.graded_blocks()
    p_home = np.load(args.pathcrf) if args.pathcrf else None
    graded_codes = np.full(len(trk), pm.X, dtype=np.int8)
    for _, a, b in blocks:
        graded_codes[a:b + 1] = 0
    runs = pm.valid_runs(graded_codes)

    masks = [synthetic_mask(trk, blocks, s) for s in SEEDS]
    # Real ball gaps concentrate in crowded penalty areas; score that subset separately.
    L, W = data.PITCH
    bx, by = trk[:, -1, 0] * L, trk[:, -1, 1] * W
    with np.errstate(invalid="ignore"):
        in_box = ((bx < 16.5) | (bx > L - 16.5)) & (np.abs(by - W / 2) < 20.16)
    for name in args.variants.split(","):
        (overrides, use_pcrf, use_aux), mode = parse_variant(name)
        p = dict(PARAMS, **overrides)
        for item in args.set:
            k, v = item.split("=", 1)
            p[k] = type(PARAMS[k])(float(v))
        aux = data.aux_ball(args.aux, args.aux_key, p["AUX_CONF"], int(p["AUX_MIN"])) if use_aux else None
        ph = p_home if use_pcrf else None
        mdl = model if mode else None
        enabled = [c for c in COMPONENTS if not (mode == "replace" and c in ("control", "deadball"))]
        full, info = labels_for(trk, team, tags, blocks, p, ph, aux, mdl, enabled)
        ctrl_team = np.where(info["near_team"] == data.HOME, "H", "A")
        ctrl = info["ctrl"] & (info["source"] == 1)
        res = {"team_acc": [], "team_acc_box": [], "alive_acc": [], "self_spell": []}
        for mask in masks:
            masked = trk.copy()
            masked[mask, -1, :] = np.nan
            out, _ = labels_for(masked, team, tags, blocks, p, ph, aux, mdl, enabled)
            m = mask & ctrl
            res["team_acc"].append(float((out[m] == ctrl_team[m]).mean()))
            mb = m & in_box
            res["team_acc_box"].append(float((out[mb] == ctrl_team[mb]).mean()))
            res["alive_acc"].append(float(((out[mask] == "D") == (full[mask] == "D")).mean()))
            gt = pm.label_codes(full)
            pr = pm.label_codes(out)
            res["self_spell"].append(pm.spell_scores(gt, pr, runs)["mean_f1"])
        summary = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
        summary.update({k + "_sd": round(float(np.std(v)), 4) for k, v in res.items()})
        print(json.dumps({"variant": name, **summary}), flush=True)


if __name__ == "__main__":
    main()
