# More than a Hack 2027 — Team Saargart: technical report

Code and reproduction: https://github.com/leemingo/mtah2027-possession (`make_submission.sh LP0B05`).

## 1. Approach in one paragraph

We treat possession as a **segmentation** problem, not a per-frame classification. For every frame
we add up evidence scores for the three states H / A / D from:

- (i) ball control in the tracking
- (ii) dead-ball cues
- (iii) Smart Tagging
- (iv) a frame classifier trained on **other** matches (Sportec Open DFL)
- (v) a ball-free team prior from the pretrained PathCRF model, inside ball gaps
- (vi) a video ball detector, where the provided ball track is missing

A 3-state HMM decoded with Viterbi, one graded half at a time, turns the scores into labels. Each
change of state has a fixed cost, so a touch or a deflection does not change possession, while
sustained control by the other team does. The change is placed where that control starts. No
parameter or model was fitted to labels of this match.

## 2. Possession rules implemented

| rule | implementation (`possession/evidence.py`, `params.py`) |
|---|---|
| controlled possession | nearest player of a team within 1.5 m, ball slow or moving with the player, sustained ≥ 0.2 s → strong vote for that team |
| contested / loose ball | both teams within 2 m → weak vote for the nearer team; nobody close → no vote, so possession stays with the last team |
| passes, deflections, aerial duels | ball faster than 8 m/s with nobody within 2 m → no team vote; a team change costs about 0.5 s of clean opponent control |
| ball out of play | ball beyond the lines by more than 0.5 m, stationary ball ≥ 1 s, low collective player speed (2 s window, 1.5 m/s), kickoff formation |
| restarts | Smart Tagging SET PIECES: dead before the restart, the tagged team after it. The tag window opens about 3.5 s before the restart, measured on the tagged kickoffs and on the player-speed profile around every set-piece tag. Throw-ins (Team = N/A) give only the dead-ball part |
| start of a half | every half starts with a kickoff, so frames before the half's tagged *Kick Off (Start)* restart are dead |
| ball not tracked (41 % of graded frames, median gap 10.7 s) | no rule vote, so the previous state persists unless other evidence builds up. The learned classifier, Smart Tagging phases (offensive → tagged team, defensive → opponent), PathCRF and the video ball vote there |
| abstention | never `X` on graded frames; `X` outside the graded halves. spellF1 forward-fills `X` anyway, and macroF1 would charge it as a miss |

## 3. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | core input. Slots are tracklets, so all geometry is team-level (team-wise nearest distance). The speed channel is in m/s: the XML says kph, but position differences give a ratio of 1.00 |
| Smart Tagging | set-piece restarts, half starts, phase-based team prior |
| tactical video | WASB zero-shot ball detection (`ball_detection/`, linked into a track by `link.py`). Used only where the Metrica ball is missing |
| homography | only inside the ball detector (pixels → pitch) |
| public 5-minute ground truth | only to check that the pipeline is correct (alignment, label meaning); never used to choose parameters, models or variants |

## 4. Training data and pretrained models

**Nothing from the provided game, and no labels of any kind on it.**

- **Learned emission (ours):** a histogram gradient-boosting classifier, `possession/learned.py`,
  with 36 team-symmetric features.
  - Features: distances and counts around the ball, ball speed and direction change, relative
    speed, pitch position, collective motion, and temporal context including ball-gap context.
  - Trained on the 7 **Sportec Open DFL** matches (Bassek et al., *Scientific Data* 2025; CC BY).
  - Labels: D where the provider's `BallStatus` = 0, otherwise the team in the provider's
    `BallPossession`. These come from the DFL's official data provider for those games.
  - Each training match is used three times: clean, with the ball removed on 41 % of the frames in
    long blocks (lognormal lengths, median 268 frames, as in our match), and with the ball and
    about 0.8 player per team removed.
  - Size: 602k frames (every 5th frame).
  - The model is committed (`possession/models/learned_dfl.pkl`).
  - `possession/training/train_dfl.sh` downloads the data (md5-checked), parses it and retrains.
- **Tried and dropped: SkillCorner open data.** We tested the A-League 2024/25 set (20 matches,
  MIT): D from its Phases-of-Play boundaries, H/A from its per-frame possession group, untracked
  replay frames excluded. Adding it to the training data made things worse on both checks below:
  held-out DFL spellF1 dropped by 0.027 (worse in 7 of 7 matches), and masked accuracy on our match
  fell from 0.828 to 0.807. The more SkillCorner matches we added, the worse it got (5 / 10 / 20
  matches: −0.014 / −0.020 / −0.026). Three reasons, measured:
  - **different dead-ball definition:** its D (outside Phases of Play) includes live-looking play.
    A DFL-trained model calls only 1 % of SkillCorner D frames dead, and adding SkillCorner raised
    the predicted D share on our match from 0.33 to 0.41;
  - **coarser possessions:** median spell 5.6 s against 3.0–4.7 s on DFL;
  - **noisier tracking:** broadcast-based, with 43 % extrapolated player positions; its label
    agrees with the nearest team in 85 % of controlled frames, against 92 % on DFL.
- **Tried and kept: the classifier family.** With matched hyper-parameters, LightGBM, CatBoost and
  XGBoost score within 0.004 spellF1 of histogram gradient boosting on held-out DFL (ball + player
  dropout: 0.395 / 0.393 / 0.392 vs 0.396); a random forest is worse (0.369, 7 of 7 matches) and
  five times slower. We kept scikit-learn's model (no extra dependency).
- **PathCRF** (Kim et al., KDD 2026; github.com/hyunsungkim-ds/pathcrf, MPL-2.0): checkpoint
  `saved/140`, trained by its authors on Sportec Open DFL, used unmodified.
- **WASB** (Tarashima et al., BMVC 2023; github.com/nttcom/WASB-SBDT, MIT): public soccer weights
  trained on ISSIA-CNR, used zero-shot.

## 5. Variants and label-free selection

All variants share the decoder and differ only in their evidence:

| variant | evidence |
|---|---|
| A | rules + Smart Tagging |
| P0 | A + PathCRF P(home \| alive), only where the ball is missing, tempered towards ½ (ε = 0.35) |
| B05 | A + WASB zero-shot ball where the Metrica ball is missing, kept only in runs of ≥ 0.5 s with confidence ≥ 0.5, weighted ½ |
| P0B05 | A + PathCRF + WASB |
| **LP0B05** | P0B05 + the learned emission (weight 0.5) |

The learned evidence is larger than the rule evidence. On sustained-control frames the per-frame
margin is 1.00 for the rules and 1.30 for the learned term. To keep the same "about 0.5 s of clean
control per change", all three transition costs are scaled from 12 to 27.5 (= 12 × 2.3).
Scaling only the team-change cost would break `C_SWITCH ≤ C_DEAD + C_RESTART`: the decoder would
then route team changes through short dead-ball detours.

**How we select without labels.**

- **Masked-ball accuracy:** we hide the Metrica ball on 15 % of the ball-visible graded frames, in
  gaps whose lengths are resampled from the real gap distribution, with 3 seeds. We rerun each
  variant and measure, on the masked frames, how often it recovers the team that sustained ball
  control shows on the unmasked data (`possession/maskeval.py`).
- **Box subset:** the same measure restricted to masked frames with the ball in a penalty area,
  where most real gaps are.
- **Guards:**
  - the file validates
  - `ctrl_agree` ≥ 0.9, i.e. the output names the controlling team on sustained-control frames
  - ≤ 10 % of possessions shorter than 2 s
  - the md5 reproduces from a cold start
- **Ties:** within one standard deviation, the variant with fewer dependencies wins.

| variant | masked team acc. (±sd) | in the box | dead/alive agreement | possessions | < 2 s | ctrl_agree |
|---|---|---|---|---|---|---|
| A | 0.664 (0.084) | 0.318 | 0.922 | 230 | 5.2 % | 0.92 |
| P0 | 0.725 (0.024) | 0.406 | 0.921 | 230 | 5.2 % | 0.92 |
| B05 | 0.747 (0.025) | 0.352 | 0.917 | 231 | 4.3 % | 0.90 |
| P0B05 | 0.789 (0.009) | 0.482 | 0.917 | 230 | 4.3 % | 0.90 |
| **LP0B05** | **0.828 (0.018)** | **0.775** | **0.926** | 277 | 2.9 % | 0.95 |

Real-gap plausibility of the video ball was checked against two independent proxies: PathCRF, and
a "bridge" proxy (the same controlling team just before and just after the gap). Agreement for
the control team where only the WASB ball is available:

| WASB confidence | vs PathCRF | vs bridge |
|---|---|---|
| ≥ 0.7 | 87 % | 100 % |
| ≥ 0.5 | 74 % | 87 % |
| ≥ 0.3 | 63 % | 73 % |

That is why the gate is 0.5. Smart Tagging was not used as a criterion: inside the ball gaps the
box-related phase codes are the least reliable source we have (58–64 % on ball-visible control
frames, against 82 % for PathCRF).

**Chosen: LP0B05.** Fallbacks: **P0B05** (no learned model) and **P0** (CPU only).
A CPU-only learned variant (LP0) did not beat P0: masked accuracy 0.721 vs 0.725, and worse
dead/alive agreement.

## 6. External validation on Sportec Open DFL

Real spellF1 with the official metric code, on the 7 DFL matches. Labels are as in section 4.
There is no Smart Tagging on DFL, so the rule components are ball control and dead ball only.
The learned model is evaluated **leave-one-match-out**: trained on 6 matches, tested on the
seventh. Three test conditions: clean; ball dropout (41 % in long blocks); ball + player dropout.
Mean over 7 matches:

| configuration | clean | ball dropout | ball + player dropout |
|---|---|---|---|
| organisers' baseline (nearest player, ±50 vote) | 0.308 | 0.210 | 0.187 |
| rules, `V_MID` 1.2 | 0.449 | 0.297 | 0.286 |
| rules, `V_MID` 1.5 | 0.442 | 0.308 | 0.296 |
| **rules + learned, costs 27.5 (submitted setting)** | 0.516 | **0.406** | **0.396** |
| rules + learned, costs 12 | 0.561 | 0.441 | 0.430 |
| rules + learned (+ SkillCorner in training), costs 27.5 | 0.485 | 0.379 | 0.370 |

- **Rules vs the baseline:** the rules beat it in every match and condition, by 0.09 to 0.14.
- **Learned emission:** rules + learned beat the rules alone in 7 of 7 matches in every condition
  (smallest gain 0.05 to 0.08). The classifier's own frame accuracy is 0.85 under ball dropout.
- **`V_MID`:** 1.5 is better under ball dropout in 7 of 7 matches. Dead-ball detection is defined
  the same way in both datasets, so we adopted it.
- **Transition costs:** lower costs score higher on DFL. But DFL counts possession at touch level:
  30 % of its possessions are shorter than 2 s and the median is 3.6 s. The challenge explicitly
  does not count touches, deflections or duels without control. We therefore keep the costs at
  their "about 0.5 s of clean control" value and do not transfer the DFL optimum.

### 6.1 Second external check: SkillCorner open data

The same protocol on the 20 SkillCorner A-League matches, which were never used for training
(labels as in section 4; they are SkillCorner's algorithmic labels, so absolute values are not
comparable with DFL):

| configuration | clean | ball dropout | ball + player dropout |
|---|---|---|---|
| organisers' baseline | 0.289 | 0.205 | 0.187 |
| rules | 0.372 | 0.304 | 0.299 |
| **rules + learned (submitted setting)** | 0.350 | **0.327** | **0.322** |

The submission beats the baseline in 20 of 20 matches in every condition, and the rules in 18 of
20 under ball dropout. The learned term helps team decisions inside ball gaps (team accuracy 0.76 →
0.80) but hurts dead-ball detection on this dataset (D-F1 0.77 → 0.57), see section 7.

### 6.2 Dead-ball experiments (not adopted)

DFL tracking freezes the ball position whenever the ball is dead (99.5 % of D frames have a
static ball; live play 0.1 %). SkillCorner and our match have no frozen ball. A classifier trained
on DFL can therefore learn "static ball = dead", which does not transfer. We tried three fixes,
each scored on DFL, on DFL with the frozen ball removed, on SkillCorner and with the masked-ball
test on our match:

| option | DFL | DFL, no frozen ball | SkillCorner | our match (masked acc.) |
|---|---|---|---|---|
| submitted | 0.396 | 0.347 | 0.327 | 0.828 |
| + stoppage detector as dead-ball evidence | 0.396 | 0.354 | 0.329 | 0.816 |
| classifier for home/away only, dead ball from the rules | 0.362 | 0.320 | 0.360 | 0.793 (fails `ctrl_agree`) |
| classifier retrained without the frozen ball | 0.305 | 0.379 | 0.322 | 0.818 |

(DFL: ball + player dropout; SkillCorner: ball dropout.) The stoppage detector (a team member's
work: 3 s median of the mean player speed, walking-pace hysteresis, ≥ 6 s) finds stoppages we
miss, but on SkillCorner 45 % of the dead frames it adds beyond ours are live play, so gains and
losses cancel. No option improved on all checks, so the submission is unchanged.

## 7. Limitations

- **Ball gaps:** they are long and concentrated in crowded situations. The learned emission
  raised masked accuracy in the box from 0.48 to 0.78, but it remains the weakest situation.
- **Domain shift:** the learned model and PathCRF were trained on clean Bundesliga tracking with
  stable identities. Our input is fragmented automatic tracking, partly mimicked by dropout during
  training. PathCRF runs on pseudo-players rebuilt from tracklets, so it is tempered and weighted
  low.
- **Transition costs:** the cost scale sets the granularity of possessions. It is a prior derived
  from the challenge's definition; we cannot validate it against this challenge's labels.
- **Dead ball:** the weakest part. Detection relies on collective motion and the classifier, whose
  dead-ball knowledge partly rests on DFL's frozen ball (section 6.2). On SkillCorner our dead-ball
  recall is 0.45. Replacing our dead-ball labels by the true ones would raise SkillCorner spellF1
  from 0.33 to 0.45, so a more precise stoppage detector (for example one that requires a second
  cue such as a set-piece tag or a missing ball) is the most promising next step.
- **Proxies:** masked-ball accuracy and DFL spellF1 rank variants, but they are proxies. They do
  not predict the leaderboard score.

## 8. Reproduction

```bash
# data/ as in DATA.md; three environments:
pip install -r requirements-possession.txt                                   # Python 3.10
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu \
  && pip install -r requirements-pathcrf.txt                                 # Python 3.11
pip install -r ball_detection/requirements.txt                               # Python 3.12, GPU
PY_POSS=... PY_PCRF=... PY_BALL=... bash make_submission.sh LP0B05 Saargart.csv
# optional: retrain the learned model (downloads ~2.6 GB of Sportec Open DFL)
PY_DFL=... PY_POSS=... bash possession/training/train_dfl.sh
```

Verified:

- **Cold start:** A, P0 and LP0B05 reproduce bit-identically from fresh environments with an empty
  cache and PathCRF re-cloned at the pinned commit. `make_submission.sh` regenerates A, P0, P0B05
  and LP0B05 with identical md5. The PathCRF adapter is deterministic on CPU.
- **WASB across GPUs:** inference is bit-identical across runs on the same GPU, but not across GPUs.
  Our track and our teammate's (Kaggle GPU) differ on 8 of 119k frames, and by up to 4 m in a few
  positions. **The submission does not change:** files built from either track have the same md5,
  because the confidence/duration gate and the HMM absorb that noise.
- **Learned model:** retraining with the same thread count gives a byte-identical model file. With
  a different thread count the file differs, but the submission is unchanged (0 frames differ).
- **Runtime:**
  - possession step: about 10 s
  - PathCRF adapter: about 7 min on 8 CPU threads
  - WASB: about 1.5 h per half on one shared A6000
  - `link.py`: about 10 min
  - model training: about 1 min
