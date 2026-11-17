# Rule-based ball possession: technical document

More Than A Hack 2027, Phase 1. Team: N-deed.

## 1. Approach

A deterministic, rule-based system. No model is trained and no parameter is fitted. Every
threshold is set in advance from football knowledge and from the possession definition in
`CHALLENGE.md`, then left unchanged.

We chose rules over a learned model for three reasons:

- The rules forbid training or tuning on the provided game, and no labelled possession data
  from other games was available to us in the time we had.
- The definition of possession is itself a set of rules (controlled possession, loose balls stay
  with the last team, deflections do not count). Encoding them directly keeps every decision
  explainable to a coach or an analyst.
- The ranking metric (spellF1) rewards whole, stable possessions. A state machine with
  hysteresis produces that structure by construction, where a per-frame classifier would need
  heavy smoothing afterwards.

The pipeline has three stages.

**Stage 1. Ball in play or dead (D).** A frame is dead when any of these holds:

1. The ball is observed clearly beyond a touchline or goal line (more than 0.5 m, on at least
   3 observations within 2 s). It stays dead until it is seen back on the pitch and struck
   (speed above 4 m/s). This covers throw-ins, corners, goal kicks and goals.
2. The ball is stationary (under 1 m/s) for at least 3 s while the players are walking (mean
   player speed under 1.6 m/s over a 3 s window). This covers fouls and free kicks being set up.
   The 3 s floor keeps a goalkeeper holding the ball in play.
3. The ball is not tracked for at least 4 s while the players are walking. This covers injuries,
   substitutions and other long stoppages.
4. A Smart Tagging `SET PIECES` instance: the stretch before the ball is struck inside the tagged
   window is dead, extended backwards (up to 8 s) while the players are walking or the ball is
   not tracked.

The mask is then cleaned: alive gaps under 1 s inside a stoppage are filled, and dead stretches
under 2 s are discarded. A real stoppage is never that short, and a spurious D inside a
possession would split it in two.

**Stage 2. Control evidence, per frame.** The team of the player nearest the ball, only if that
player is within 2 m of the ball and at least 0.5 m closer than the nearest opponent. Otherwise
(ball not tracked, loose, or contested) there is no evidence.

**Stage 3. Possession state machine, inside each alive stretch.**

- A loose or contested ball keeps the last team in possession, as `CHALLENGE.md` specifies.
- The opponents take over only after about 0.5 s (13 frames) of cumulated control evidence, from
  one or several of their players, with no touch by the team in possession in between and no
  gap over 2 s. A simple touch, a deflection or a lost aerial duel stays below that.
- Once a change is confirmed, it is dated back to the opponents' first touch, which is where
  control was actually gained.
- After a restart, the team named by the Smart Tagging set piece starts in possession when it
  names one (it does not for throw-ins). Otherwise, the first team with 0.2 s of evidence owns
  the ball from the restart.
- An alive stretch with no evidence at all inherits the last team in possession.

Frames outside the two graded halves are written as X.

## 2. Assets used

| asset | used | how |
|---|---|---|
| Automatic tracking (ATD), players and ball | yes | positions, player speeds, team of each track slot |
| Smart Tagging | yes | `SET PIECES` only: restart timing and restart team |
| Homography | no | not needed: the ATD is already in pitch coordinates |
| Tactical and broadcast videos | no | not used as input, never used to check the output |
| Broadcast ATD | no | not synchronised to the tactical timeline |

Other Smart Tagging phases (Build Up, Progression, etc.) were deliberately left out. They are
phases, not possessions, and copying their intervals would import their errors.

## 3. Training data

None. No labelled dataset of any kind was used, and no parameter was fitted on any data.

The public 5-minute ground truth was used once, to check that the pipeline was correct (frame
alignment, file format, labels meaning what we think). It was not used to choose or change any
threshold.

## 4. Possession rules implemented

| situation | decision |
|---|---|
| loose ball, ball in the air, ball not tracked | possession stays with the last team |
| contested ball (both teams within 0.5 m of each other's distance) | no change |
| a touch or deflection by the opponents (under 0.5 s of control) | no change |
| an aerial duel won without control afterwards | no change |
| the opponents control the ball for about 0.5 s, or chain touches between their players | change, dated back to their first touch |
| ball over a line | dead until the restart is struck |
| ball placed and players walking (foul, free kick) | dead |
| restart | the tagged restart team, or the first team to control the ball |

## 5. Results

Public 5-minute ground truth, scored with the official `evaluation/run_evaluation.py`:

| submission | macroF1 | spellF1 | possessions (truth: 17) |
|---|---|---|---|
| rules_v1 (this work) | 0.7421 | 0.4667 | 19 |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 |

These 17 possessions are too few to rank systems reliably, as `EVALUATION.md` shows. The
figures are reported as a correctness check, not as an estimate of the full-match score.

## 6. Limitations and what we would do with more time

- **Changes of possession during a tracking gap** are detected late, when the ball reappears.
  Interpolating the ball through short gaps, or inferring the possessor from the collective shape
  (which team is advancing, which team is pressing), would recover part of them.
- **Quick restarts** with no visible pause, and stoppages where the players keep moving, are
  missed by the dead-ball rules.
- **Long balls and aerial duels**: the fixed 0.5 s control rule cannot always separate a
  controlled header from a flick-on.
- **Fixed thresholds**: with more time, we would calibrate them on external tracking data with
  public event annotations (for example Metrica's open sample games), which the rules allow.
- **Tracking noise**: identity switches do not matter here because only the team is used, but
  duplicated or missing players can bias the nearest-player test near the ball.

## 7. Reproduction

Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r solution/requirements.txt
# put the data bundle in ./data (see DATA.md), at least data/tactical/:
#   BAR-ATH_tactical_atd.txt, BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_smart_tagging.xml
python solution/possession.py
python evaluation/validate_submission.py evaluation/results/rules_v1.csv
python evaluation/run_evaluation.py
```

`solution/possession.py` is deterministic: the same data produces the same file, byte for byte.
It runs in under a minute on a laptop.
