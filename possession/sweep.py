"""Label-free sensitivity sweep: vary one parameter at a time around the defaults and print the
output's own statistics. No ground truth is read. Used to check that the defaults sit on a plateau
rather than to optimise anything.

    python -m possession.sweep
"""

import json
import os
import tempfile

from . import data, decode
from .params import COMPONENTS, PARAMS
from .run import build_scores, diagnostics

GRID = {
    "C_SWITCH": [6, 9, 12, 18, 24],
    "C_DEAD": [6, 12, 24],
    "R_CTRL": [1.0, 1.5, 2.0],
    "MIN_CTRL": [3, 5, 10],
    "W_MOTION": [0.15, 0.3, 0.45],
    "V_MID": [0.9, 1.2, 1.5],
    "W_TAG": [0.0, 0.025, 0.05, 0.1],
}
KEYS = ("possessions", "dur_median_s", "share_lt2s", "D_share", "dead_segments", "ctrl_agree",
        "team_switches", "switch_in_gap")


def main():
    trk = data.load_tracking()
    team = data.slot_teams()
    tags = data.smart_tags()
    blocks = data.graded_blocks()
    for name, values in GRID.items():
        for value in values:
            p = dict(PARAMS)
            p[name] = type(PARAMS[name])(value)
            S, info = build_scores(trk, team, tags, p, list(COMPONENTS))
            labels = decode.decode(S, blocks, decode.transition_costs(p), len(trk))
            d = diagnostics(labels, blocks, info)
            mark = " *" if value == PARAMS[name] else ""
            print("{:<10} {:>6}{:<2} ".format(name, value, mark)
                  + "  ".join("{}={}".format(k, d[k]) for k in KEYS), flush=True)


if __name__ == "__main__":
    main()
