# possession/ — team possession from tracking + Smart Tagging

Regenerates the submission from the provided data (`data/tactical/…`, see `../DATA.md`).
numpy and pandas only; deterministic (same md5 on every run).

```bash
pip install -r requirements-possession.txt          # numpy==2.2.6, pandas==2.3.3 (Python 3.10)
python -m possession.run --out evaluation/results/A.csv
python evaluation/validate_submission.py evaluation/results/A.csv
```

The first run parses the raw ATD into `data/cache/tactical_atd.npy` (gitignored, ~5 s).

The submitted variant (LP0B05) adds three optional inputs; `../make_submission.sh` runs everything:

```bash
bash make_submission.sh A     out.csv    # this module only
bash make_submission.sh P0    out.csv    # + PathCRF prior   (pathcrf_adapter.py, CPU torch)
bash make_submission.sh P0B05 out.csv    # + WASB video ball (ball_detection/, GPU)
bash make_submission.sh LP0B05 out.csv   # + frame classifier trained on Sportec Open DFL  <- submitted
```

| option | input | where it is used |
|---|---|---|
| `--pathcrf FILE` | per-frame P(home \| alive) from `pathcrf_adapter.py` (pretrained PathCRF, unmodified) | ball-missing frames only, tempered (`PCRF_EPS`), weighted by its label-free reliability |
| `--aux FILE --set AUX_CONF=0.5` | WASB zero-shot ball track (`ball_detection/src/link.py`) | Metrica-ball-missing frames only, ≥ 0.5 s runs with confidence ≥ 0.5, weight ½ |
| `--model FILE --set W_L=0.5` | frame classifier P(H, A, D) from `learned.py`, trained on other matches only (`models/learned_dfl.pkl`, Sportec Open DFL with simulated ball/player dropout; retrain with `training/train_dfl.sh`) | every frame, added to the rule evidence; transition costs scaled 12 → 27.5 so a change still needs ≈ 0.5 s of clean control |

Keep `C_SWITCH ≤ C_DEAD + C_RESTART`: otherwise the decoder routes team changes through short
dead-ball detours.

## Method

Per-frame evidence scores for H / A / D, decoded with a 3-state HMM (Viterbi), one graded half at
a time. All parameters are physical / football priors in `params.py`; none was fitted to labels.

| component | rule | file |
|---|---|---|
| ball control | team-wise nearest distance; *control* = nearest player within 1.5 m, ball slow or moving with the player, for ≥ 0.2 s → strong vote; contested / near → weak vote; ball in flight (> 8 m/s, nobody within 2 m) → no team vote | `evidence.control_evidence` |
| dead ball | ball beyond the lines, stationary ball, low mean player speed (2 s window), kickoff formation | `evidence.deadball_evidence` |
| set pieces | Smart Tagging SET PIECES: dead before the restart (window opens ~3.5 s before it), tagged team after it | `evidence.setpiece_evidence` |
| half start | every half starts with a kickoff: frames before the tagged *Kick Off (Start)* restart are dead | `evidence.period_start_evidence` |
| phase tags | offensive phases vote for the tagged team, defensive phases for the opponent (weak; matters mostly where the ball is missing) | `evidence.tag_evidence` |
| decoding | changing team costs `C_SWITCH` (~0.5 s of clean control by the other team), so touches and deflections do not change possession; the change is placed where the new control starts | `decode.py` |

Where the ball is not tracked (41 % of graded frames, in long blocks) there is no ball vote, so the
previous state persists unless dead-ball or tag evidence accumulates.

## Tools (label-free)

```bash
python -m possession.maskeval --variants A,P0,P0B05,L+:P0B05 --pathcrf data/cache/pathcrf_p_home.npy \
    --aux data/ball/track_wasb_full.npz          # model selection: mask the ball where it is visible
python -m possession.sweep                       # one-at-a-time sensitivity of the output's own statistics
python -m possession.run --out x.csv --check-public   # public 5 min: pipeline sanity check only
```

The public 5-minute ground truth is never used to choose parameters or variants (RULES.md).
