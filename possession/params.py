"""All parameters in one place. Values are physical / football priors, not fitted to any labels.

Distances in metres, speeds in m/s, durations in frames (25 fps). Scores are per-frame votes;
costs are one-off penalties for a change of state (see decode.py).
"""

PARAMS = {
    # 1) ball control
    "R_CTRL": 1.5,          # control radius
    "V_CTRL_ABS": 3.0,      # ball this slow next to a player counts as controlled ...
    "V_CTRL_REL": 2.5,      # ... or moving at the player's own speed (dribble)
    "MIN_CTRL": 5,          # 0.2 s of sustained control by one team
    "W_CTRL": 1.0,
    "R_CONT": 2.0,          # contested: both teams this close
    "W_CONT": 0.3,
    "R_NEAR": 3.0,
    "W_NEAR": 0.15,
    "V_FLIGHT": 8.0,        # ball in flight: no team vote
    "W_FLIGHT_ALIVE": 0.2,
    "V_STILL": 0.5,         # stationary ball (set-piece setup) ...
    "STILL_MIN": 25,        # ... for at least 1 s
    "AUX_WEIGHT": 0.5,      # auxiliary (video-detected) ball counts half

    # 2) dead ball
    "M_OUT": 0.5,           # beyond the lines by this much
    "W_OUT": 2.0,
    "W_STILL": 0.5,
    "MOTION_WIN": 51,       # 2 s centred window for the mean player speed
    "V_MID": 1.5,           # below: stoppage-like, above: play-like. 1.2 -> 1.5 after external
                            # validation on Sportec Open DFL with simulated ball dropout
                            # (+0.011 spellF1, +0.015 macroF1, 7/7 matches)
    "V_SCALE": 0.4,
    "W_MOTION": 0.3,
    "KO_MARGIN": -1.0,      # players may stand up to 1 m over the halfway line
    "KO_TOL": 1,            # ... and one per team may be on the wrong side (the kicker)
    "KO_MIN_PLAYERS": 8,
    "KO_MIN_LEN": 25,
    "W_KO": 1.0,

    # 3) set pieces (Smart Tagging)
    "SP_OFFSET": 88,        # tag window opens ~3.5 s before the restart
    "SP_PRE": 125,          # dead-ball vote from 5 s before the window opens
    "W_SP_DEAD": 0.5,
    "SP_POST": 125,         # 5 s after the restart: alive, tagged team
    "W_SP_ALIVE": 0.3,
    "W_SP_TEAM": 0.2,
    "W_PRESTART": 5.0,      # before a half's kickoff: dead ball

    # 4) Smart Tagging phases
    "W_TAG": 0.05,
    "TAG_CLIP": 2.0,
    "TAG_RELIABILITY": 0,   # 1: scale each phase code by its label-free reliability
    "REL_REF_ACC": 0.75,    # reliability that maps to weight 1
    "TAG_REL_MIN": 0.0,     # codes less reliable than this are dropped (0 = keep all)

    # 7) learned frame classifier (only with --model)
    "W_L": 1.0,
    "L_EPS": 0.1,

    # 6) PathCRF (only with --pathcrf)
    "W_PCRF": 0.05,
    "PCRF_EPS": 0.35,       # tempering towards 1/2 (raw output is overconfident)

    # HMM transition costs
    "C_SWITCH": 12.0,       # ~0.5 s of clean control by the other team
    "C_DEAD": 12.0,
    "C_RESTART": 12.0,

    # auxiliary ball gating (only with --aux)
    "AUX_CONF": 0.7,
    "AUX_MIN": 13,          # 0.5 s
}

COMPONENTS = ("control", "deadball", "setpiece", "tags")
