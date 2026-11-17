# Rule-based ball possession: technical document

More Than A Hack 2027, Phase 1. Team: N-deed.

## 1. Approach

A deterministic, rule-based system. No model is trained. Version 1 used thresholds set from
football knowledge alone. Version 2 (the submitted one) recalibrates a few of them on external
data only, Metrica's open Sample Game 3 (section 3). Nothing is fitted on the challenge match.

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
   player speed under 1.8 m/s over a 3 s window). This covers fouls and free kicks being set up.
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
player is within 1.5 m of the ball and at least 0.5 m closer than the nearest opponent. Otherwise
(ball not tracked, loose, or contested) there is no evidence.

**Stage 3. Possession state machine, inside each alive stretch.**

- A loose or contested ball keeps the last team in possession, as `CHALLENGE.md` specifies.
- The opponents take over only after about 0.3 s (8 frames) of cumulated control evidence, from
  one or several of their players, with no touch by the team in possession in between and no
  gap over 2 s. A simple touch, a deflection or a lost aerial duel stays below that.
- Once a change is confirmed, it is dated back to the opponents' first touch, which is where
  control was actually gained. If the ball was not tracked for most of the time between the last
  touch of the team in possession and that first touch, the change is dated to the middle of
  the gap instead of the moment the ball reappears.
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
| Metrica Sample Game 3 (external) | yes | calibration only, see section 3 |

Other Smart Tagging phases (Build Up, Progression, etc.) were deliberately left out. They are
phases, not possessions, and copying their intervals would import their errors.

## 3. Calibration data (external only)

| dataset | source | labels | who labelled it | size |
|---|---|---|---|---|
| Sample Game 3 | Metrica Sports open data, github.com/metrica-sports/sample-data | event data (passes, carries, recoveries, set pieces, balls out) | Metrica Sports, published with the data | one full anonymised match, 143,761 frames, 3,620 events |

No data from the challenge match was used to set any parameter.

**Reference.** We derive frame labels from the events: dead from the end of the last event before
each set piece until the set piece; otherwise the team of the most recent on-ball event (pass,
carry, recovery, set piece, shot, ball lost; challenges ignored). This gives 404 possessions.

**Degradation.** Sample Game 3 tracking is cleaner than the challenge ATD, so before running our
rules we remove the ball until 41% of frames have none (gaps of random length, mean 2 s), drop 5%
of player observations and add 0.3 m of position noise. Player speeds are computed from positions
over 5 frames. Two random seeds, fixed.

**Search.** A coarse grid over the walking threshold, stoppage durations, control radius, contest
margin and confirmation time, scored with the official `possession_metrics.py`. We picked a
setting in a flat region of the grid rather than the single best point, and chose 1.8 m/s rather
than 2.0 m/s for the walking threshold to stay conservative where the real tracking loses the
ball more often than our simulation.

| version | spellF1 on Sample Game 3 (degraded) | macroF1 |
|---|---|---|
| v1, a priori thresholds | 0.528 | 0.820 |
| v2, calibrated | 0.565 | 0.833 |

The public 5-minute ground truth was used only to check the pipeline end to end (frame
alignment, file format). It was not used to choose or change any threshold.

`solution/calibrate_metrica.py` reproduces the calibration.

## 4. Possession rules implemented

| situation | decision |
|---|---|
| loose ball, ball in the air, ball not tracked | possession stays with the last team |
| contested ball (both teams within 0.5 m of each other's distance) | no change |
| a touch or deflection by the opponents (under 0.3 s of control) | no change |
| an aerial duel won without control afterwards | no change |
| the opponents control the ball for about 0.3 s, or chain touches between their players | change, dated back to their first touch |
| ball over a line | dead until the restart is struck |
| ball placed and players walking (foul, free kick) | dead |
| restart | the tagged restart team, or the first team to control the ball |

## 5. Results

Public 5-minute ground truth, scored with the official `evaluation/run_evaluation.py`:

| submission | macroF1 | spellF1 | possessions (truth: 17) |
|---|---|---|---|
| rules_v2 (this work, submitted) | 0.7416 | 0.4444 | 19 |
| rules_v1 (a priori thresholds) | 0.7421 | 0.4667 | 19 |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 |

These 17 possessions are too few to rank systems reliably, as `EVALUATION.md` shows. The gap
between v1 and v2 here is one possession matched at IoU 0.5 (9 against 8), well inside the noise.
We submit v2 because its thresholds are the only ones backed by a measurement (section 3), and
we did not change anything after seeing this score.

## 6. Limitations and what we would do with more time

- **Changes of possession during a tracking gap** are detected late, when the ball reappears.
  Interpolating the ball through short gaps, or inferring the possessor from the collective shape
  (which team is advancing, which team is pressing), would recover part of them.
- **Quick restarts** with no visible pause, and stoppages where the players keep moving, are
  missed by the dead-ball rules.
- **Long balls and aerial duels**: the fixed 0.5 s control rule cannot always separate a
  controlled header from a flick-on.
- **Calibration on one external game**: the event-derived reference is an approximation of the
  challenge definition, and our degradation only imitates the real tracking gaps. More external
  games, and a gap model measured on the real ATD, would make the thresholds more reliable.
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
python solution/possession.py          # writes evaluation/results/rules_v2.csv
python evaluation/validate_submission.py evaluation/results/rules_v2.csv
python evaluation/run_evaluation.py
```

`solution/possession.py` is deterministic: the same data produces the same file, byte for byte.
It runs in under a minute on a laptop.
