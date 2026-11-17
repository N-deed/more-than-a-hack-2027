# Rule-based ball possession: technical document

More Than A Hack 2027, Phase 1. Team: N_DEED.

## 1. Approach

A deterministic, rule-based system. No model is trained. Version 1 used thresholds set from
football knowledge alone. Version 2 recalibrates a few of them on external data only, Metrica's open Sample Game 3
(section 3). Version 3, the submitted one, is version 2 plus one published match fact: the second
half is graded from video frame 110,808 but kicks off at 113,742 (FAQ.md), so those frames are
written as dead ball. Nothing is fitted on the challenge match.

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

Frames outside the two graded halves are written as X. Graded frames before the second-half
kick-off (110,808 to 113,741, a published fact) are written as D.

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
| Sample Games 1, 2, 3 | Metrica Sports open data, github.com/metrica-sports/sample-data | event data (passes, carries, recoveries, set pieces, balls out) | Metrica Sports, published with the data | three full anonymised matches, about 430,000 frames, 10,000+ events |
| 20 A-League 2024/25 matches | SkillCorner open data, github.com/SkillCorner/opendata (MIT) | phases of play with the team in possession | SkillCorner Game Intelligence, published with the data | 20 full matches of broadcast tracking at 10 fps |

The thresholds were calibrated on Sample Game 3 (the one in EPTS format, like the challenge ATD).
Games 1 and 2 were kept aside as independent checks, and used to train and test the models we
tried and dropped (section 6).

No data from the challenge match was used to set any parameter.

**Reference.** We derive frame labels from the events: dead from the end of the last event before
each set piece until the set piece; otherwise the team of the most recent on-ball event (pass,
carry, recovery, set piece, shot, ball lost; challenges ignored). This gives 404 possessions.

**Degradation.** Sample Game 3 tracking is cleaner than the challenge ATD, so before running our
rules we degrade it to look like the challenge tracking. The gap model comes from
`solution/diagnose_atd.py`, which describes the challenge ATD without using any label: the ball
is missing on 40.7% of graded frames, in only 150 gaps; no gap under 4 s holds any missing
frames; gaps of 4-15 s hold 25% of them and longer gaps (players walking, mostly stoppages) the
rest. Sample Game 3 already loses the ball during its stoppages (31.7% of frames), so we add
in-play gaps of 4-15 s until 41% of frames have no ball. We also drop 5% of player observations
and add 0.3 m of position noise. Player speeds are computed from positions over 5 frames. Two
random seeds, fixed.

**Search.** A coarse grid over the walking threshold, stoppage durations, control radius, contest
margin and confirmation time, scored with the official `possession_metrics.py`. We picked a
setting in a flat region of the grid rather than the single best point, and chose 1.8 m/s rather
than 2.0 m/s for the walking threshold to stay conservative where the real tracking loses the
ball more often than our simulation.

spellF1 on the degraded games (mean of two seeds):

| version | game 1 (not used to calibrate) | game 2 (not used to calibrate) | game 3 (calibration) |
|---|---|---|---|
| v1, a priori thresholds | 0.562 | 0.558 | 0.506 |
| **v2, calibrated (submitted)** | **0.596** | **0.578** | **0.548** |

v2 also improves on the two games it was not calibrated on, so the gain is not an artefact of
fitting game 3. On the clean, undegraded tracking of game 3, v2 reaches 0.575.

The degraded score sits close to the clean one, so on this reference the tracking gaps are not
the main source of error. What remains is the gap between geometric control and the
event-derived reference: 72% of its changes of possession are found within 2 s even on clean
tracking. Tuning further against this reference would fit its own conventions rather than the
challenge's, so we stopped here.

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
| rules_v2 | 0.7416 | 0.4444 | 19 |
| rules_v1 (a priori thresholds) | 0.7421 | 0.4667 | 19 |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 |

These 17 possessions are too few to rank systems reliably, as `EVALUATION.md` shows. The gap
between v1 and v2 here is one possession matched at IoU 0.5 (9 against 8), well inside the noise.
We kept v2's thresholds because they are the only ones backed by a measurement (section 3), and
we did not change anything after seeing this score. v3 differs from v2 only on the 2,934 frames
before the second-half kick-off, outside the public window.

## 6. What we tried and dropped

Both were tested leave-one-game-out on the three Metrica games (train on two, test on the
third), degraded with the challenge gap model. `solution/experiments.py` reproduces every number.

| attempt | idea | result (spellF1, games 1 / 2 / 3) | decision |
|---|---|---|---|
| v2 (reference) | rules | 0.596 / 0.578 / 0.548 | kept |
| team-shape model during ball gaps | a logistic model on collective features (block height, forward speed, width, length, last line, home minus away) predicts the team in possession at 77-79% per frame without the ball; used to change possession during long gaps | 0.595 / 0.578 / 0.549 at best | dropped: no gain. Only 15% of the changes of possession fall inside a ball gap, so there was little to win |
| recalibrating on SkillCorner | a second, independent reference: SkillCorner phases of play (gaps under 3 s between phases keep the previous team, longer gaps are stoppages), 20 broadcast-tracking matches resampled to 25 fps and degraded the same way; grid on 10 matches, check on the other 10 | v2: Metrica 0.574 mean, SkillCorner held-out 0.490. Best SkillCorner setting (0.8 s confirmation, 2 m, 1.4 m/s): SkillCorner 0.528 but Metrica 0.500. Best average over both providers: v2 (0.532) | v2 kept: the two providers disagree on how quickly possession changes hands, and v2 is the best compromise between them |
| whistle detector | the referee's whistle seen as a collective deceleration: mean player speed (1 s window) under 1.0-1.4 m/s for 1.5-2.5 s marks a stoppage from the start of the deceleration to the restart | overall (Metrica mean + SkillCorner held-out) 0.440 to 0.529, against 0.532 for v2 | dropped: players also slow down together during play, and the false stoppages cost more than the fouls it finds |
| last touch before a stoppage | a touch by the other team just before the ball goes dead (a block that ends in a throw-in or corner) gives them the ball from that touch | overall 0.524 against 0.532 | dropped |
| learned emissions + Viterbi | the rules replaced by a per-frame logistic model (ball distances, players near the ball, team shape) decoded with a two-state Viterbi and a switch cost | 0.478 / 0.469 / 0.475 at best | dropped: clearly worse than the rules |

The diagnosis behind these results: on the clean tracking of game 3, 28% of the reference changes
of possession are missed and 30% of predicted ones are spurious, mostly around short possessions
(under 4 s) and contested balls, where the geometric control rule and the event-based reference
disagree. Neither model fixed that.

The whistle detector and the last-touch rule came from watching the public 5 minutes next to the
video to understand the definition (allowed by the rules: fouls are dead from the whistle; a block
that ends out of play belongs to the blocking team). We implemented them as general rules,
measured them only on the external games, and dropped both. No threshold was chosen on the
public window.

The SkillCorner check makes the same point from another side: each reference prefers settings
close to its own conventions (Metrica's events change possession quickly, SkillCorner's phases
less so). The remaining error is mostly a question of definition, not of tracking quality, and
the challenge's own definition can only be read from its 5 public minutes, which the rules
forbid using for tuning.

## 7. Limitations and what we would do with more time

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

## 8. Reproduction

Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r solution/requirements.txt
# put the data bundle in ./data (see DATA.md), at least data/tactical/:
#   BAR-ATH_tactical_atd.txt, BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_smart_tagging.xml
python solution/possession.py          # writes evaluation/results/rules_v3.csv (submitted)
python evaluation/validate_submission.py evaluation/results/rules_v3.csv
python evaluation/run_evaluation.py
```

To reproduce the calibration and the experiments (external data only):

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../metrica-sample-data
python solution/diagnose_atd.py                  # gap model of the challenge ATD, no labels
python solution/calibrate_metrica.py --sample ../metrica-sample-data/data/Sample_Game_3
python solution/experiments.py --sample ../metrica-sample-data/data
python solution/experiments_skillcorner.py --skillcorner <skillcorner folder> --metrica ../metrica-sample-data/data
```

`solution/possession.py` is deterministic: the same data produces the same file, byte for byte.
It runs in under a minute on a laptop.
