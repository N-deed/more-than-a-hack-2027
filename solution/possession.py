"""Rule-based ball possession from automatic tracking + Smart Tagging.

    python solution/possession.py                       # writes evaluation/results/rules_v1.csv
    python evaluation/validate_submission.py evaluation/results/rules_v1.csv
    python evaluation/run_evaluation.py

No model, no training, no fitted parameter. Every threshold below is fixed from football
knowledge and from the challenge definition (CHALLENGE.md), never from the ground truth.

Pipeline
--------
1. Read the raw ATD once into arrays (players, their team, their speed, the ball).
2. Ball-in-play mask (D vs alive):
   a. ball beyond a touchline / goal line -> dead until the restart is played;
   b. ball stationary for several seconds while the players are walking -> dead (fouls,
      free kicks, goal kicks being placed);
   c. long stretch with no ball and no collective movement -> dead (injuries, VAR, subs);
   d. Smart Tagging SET PIECES: the part of the tagged window before the ball is struck is dead.
   Dead stretches shorter than MIN_DEAD_S are discarded: a real stoppage is never that short,
   and a spurious D in the middle of a possession splits it in two.
3. Per-frame control evidence: the team of the player closest to the ball, only if that player
   is within CONTROL_RADIUS_M and clearly closer than the nearest opponent. Ball not seen or
   loose -> no evidence.
4. Possession state machine inside each alive stretch (the "controlled possession" rule):
   - loose or contested balls keep the last team in possession;
   - the opponents take over only after CONFIRM_FRAMES frames of evidence (one or several of
     their players), uninterrupted by a touch from the team in possession; a deflection or a
     lost aerial duel does not reach that;
   - once confirmed, the change is dated back to the opponent's first touch;
   - after a restart, the first team to show RESTART_CONFIRM_FRAMES frames of evidence owns the
     ball from the restart (the Smart Tagging team is used when it names one).
5. Graded halves get H/A/D, everything else X.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Parameters (fixed a priori, see the technical document for the reasoning behind each one)
# ---------------------------------------------------------------------------------------------
FPS = 25

# Control
CONTROL_RADIUS_M = 2.0       # ball within 2 m of a player: he can play it (tracking noise ~1 m)
CONTEST_MARGIN_M = 0.5       # nearest opponent must be at least this much further away
CONFIRM_FRAMES = 13          # ~0.5 s of cumulative control before a change of possession counts
RUN_GAP_FRAMES = 2 * FPS     # evidence more than 2 s apart is not the same attempt at control
RESTART_CONFIRM_FRAMES = 5   # after a restart the first clear touch is enough (0.2 s)

# Dead ball
OUT_MARGIN_M = 0.5           # ball must be clearly over the line, not on it
OUT_MIN_FRAMES = 3           # ...for a few observations, to ignore single tracking glitches
KICK_SPEED_MS = 4.0          # a restart is played when the ball moves faster than this
STILL_SPEED_MS = 1.0         # ball "not moving"
STILL_MIN_S = 3.0            # stationary at least this long (keeper holding the ball < 3 s moves)
WALKING_MS = 1.6             # mean player speed under this = players walking, game stopped
NOBALL_MIN_S = 4.0           # no ball AND players walking for this long = stoppage
ACTIVITY_WINDOW_S = 3.0      # smoothing of the collective speed
MIN_DEAD_S = 2.0             # shorter dead stretches are noise
MIN_ALIVE_S = 1.0            # alive gaps shorter than this inside a stoppage are noise
SETPIECE_LOOKBACK_S = 8.0    # how far before a tagged set piece the stoppage may start

HOME, AWAY, NONE = 0, 1, -1

TEAM_TAG = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------
def read_meta(meta_path):
    """Team (HOME/AWAY) of each track slot. First team in the file is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def read_atd(raw_path, n_slots, n_frames, pitch):
    """Players (x, y, speed) and ball (x, y), in metres, indexed by video frame. NaN = missing."""
    length, width = pitch
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1          # frame_id counts from 1, video frames from 0
            if frame < 0 or frame >= n_frames:
                continue
            groups = parts[1].split(";")
            k = min(len(groups), n_slots)
            for slot in range(k):
                group = groups[slot]
                if not group or group[0] == "N":
                    continue
                vals = group.split(",")
                px[frame, slot] = float(vals[0]) * length
                py[frame, slot] = float(vals[1]) * width
                if len(vals) > 2 and vals[2] and vals[2][0] != "N":
                    ps[frame, slot] = float(vals[2])
            ball = parts[2]
            if ball and ball[0] != "N":
                vals = ball.split(",")
                bx[frame] = float(vals[0]) * length
                by[frame] = float(vals[1]) * width
    return px, py, ps, bx, by


def read_set_pieces(path, n_frames):
    """[(start_frame, end_frame, team or NONE)] for every SET PIECES instance."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        if (inst.findtext("code") or "").strip().lower() != "set pieces":
            continue
        start = int(round(float(inst.findtext("start")) * FPS))
        end = int(round(float(inst.findtext("end")) * FPS))
        team = NONE
        for label in inst.findall("label"):
            if (label.findtext("group") or "").strip().lower() == "team":
                team = TEAM_TAG.get((label.findtext("text") or "").strip().lower(), NONE)
        out.append((max(0, start), min(n_frames, end), team))
    return out


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def runs(mask):
    """[(start, end_exclusive)] of True runs in a boolean array."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def rolling_mean_nan(values, window):
    """Centred rolling mean ignoring NaN. NaN where the window holds no observation."""
    ok = ~np.isnan(values)
    v = np.where(ok, values, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(values))
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(len(values), idx + window // 2 + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def ball_speed(bx, by, max_gap=5):
    """Ball speed in m/s from successive observations at most max_gap frames apart."""
    n = len(bx)
    speed = np.full(n, np.nan)
    seen = np.where(~np.isnan(bx))[0]
    if len(seen) < 2:
        return speed
    gaps = np.diff(seen)
    d = np.hypot(np.diff(bx[seen]), np.diff(by[seen]))
    v = d / (gaps / FPS)
    good = gaps <= max_gap
    speed[seen[1:][good]] = v[good]
    # light smoothing: tracking jitter alone makes a still ball "move" a little
    return rolling_mean_nan(speed, 5)


# ---------------------------------------------------------------------------------------------
# Dead ball
# ---------------------------------------------------------------------------------------------
def dead_mask(bx, by, bspeed, activity, set_pieces, pitch):
    length, width = pitch
    n = len(bx)
    seen = ~np.isnan(bx)
    dead = np.zeros(n, dtype=bool)

    # a. ball out of play: from the first clear observation over a line until the restart
    #    (ball seen back on the pitch and struck)
    out = seen & ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M) |
                  (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M))
    inside = seen & ~out
    kicked = inside & (np.nan_to_num(bspeed) > KICK_SPEED_MS)
    out_idx = np.where(out)[0]
    t = 0
    while t < len(out_idx):
        start = out_idx[t]
        window = out_idx[(out_idx >= start) & (out_idx < start + 2 * FPS)]
        if len(window) < OUT_MIN_FRAMES:
            t += 1
            continue
        nxt = np.where(kicked[start:])[0]
        end = start + nxt[0] if len(nxt) else n
        dead[start:end] = True
        t = np.searchsorted(out_idx, end)

    walking = np.nan_to_num(activity, nan=0.0) < WALKING_MS

    # b. ball standing still for a while with the players walking
    still = seen & (np.nan_to_num(bspeed, nan=99.0) < STILL_SPEED_MS)
    for a, b in runs(still | ~seen):
        segment_seen = seen[a:b]
        if segment_seen.sum() == 0 or (b - a) < STILL_MIN_S * FPS:
            continue
        if still[a:b].sum() >= 0.8 * segment_seen.sum() and walking[a:b].mean() > 0.5:
            dead[a:b] = True

    # c. no ball and nobody moving
    for a, b in runs(~seen & walking):
        if (b - a) >= NOBALL_MIN_S * FPS:
            dead[a:b] = True

    # d. tagged set pieces: the stoppage runs up to the moment the ball is struck
    for start, end, _team in set_pieces:
        k = np.where(kicked[start:end])[0]
        strike = start + k[0] if len(k) else (start + end) // 2
        lo = max(0, start - int(SETPIECE_LOOKBACK_S * FPS))
        back = start
        while back > lo and (walking[back - 1] or not seen[back - 1]):
            back -= 1
        dead[back:strike] = True

    # clean: fill short alive gaps inside stoppages, then drop short stoppages
    for a, b in runs(~dead):
        if a > 0 and b < n and (b - a) < MIN_ALIVE_S * FPS:
            dead[a:b] = True
    for a, b in runs(dead):
        if (b - a) < MIN_DEAD_S * FPS:
            dead[a:b] = False
    return dead


# ---------------------------------------------------------------------------------------------
# Possession
# ---------------------------------------------------------------------------------------------
def control_evidence(px, py, team_of_slot, bx, by):
    """Per frame: team clearly in control of the ball, or NONE."""
    n = len(bx)
    ev = np.full(n, NONE, dtype=np.int8)
    dx = px - bx[:, None]
    dy = py - by[:, None]
    dist = np.hypot(dx, dy)                                  # NaN where anything is missing
    home = np.where(team_of_slot[None, :] == HOME, dist, np.nan)
    away = np.where(team_of_slot[None, :] == AWAY, dist, np.nan)
    with np.errstate(all="ignore"):
        dh = np.nanmin(np.where(np.isnan(home), np.inf, home), axis=1)
        da = np.nanmin(np.where(np.isnan(away), np.inf, away), axis=1)
    with np.errstate(invalid="ignore"):
        home_ctrl = (dh <= CONTROL_RADIUS_M) & ((da - dh >= CONTEST_MARGIN_M) | np.isinf(da))
        away_ctrl = (da <= CONTROL_RADIUS_M) & ((dh - da >= CONTEST_MARGIN_M) | np.isinf(dh))
    ev[home_ctrl] = HOME
    ev[away_ctrl] = AWAY
    ev[np.isnan(bx)] = NONE
    return ev


def possession_in_stretch(ev, a, b, restart_team, carry_team):
    """Team in possession for frames a..b-1 of one alive stretch."""
    out = np.full(b - a, NONE, dtype=np.int8)
    current = restart_team
    run_team, run_start, run_count, run_last = NONE, 0, 0, -10 ** 9

    for t in range(a, b):
        e = ev[t]
        i = t - a
        if e != NONE:
            if current == NONE:
                # restart without a tagged team: first team with a few frames of control
                if e != run_team or t - run_last > RUN_GAP_FRAMES:
                    run_team, run_start, run_count = e, t, 0
                run_count += 1
                run_last = t
                if run_count >= RESTART_CONFIRM_FRAMES:
                    current = run_team
                    out[:i + 1] = current
                    run_team, run_count = NONE, 0
            elif e == current:
                run_team, run_count = NONE, 0           # the team in possession touched it again
            else:
                if e != run_team or t - run_last > RUN_GAP_FRAMES:
                    run_team, run_start, run_count = e, t, 0
                run_count += 1
                run_last = t
                if run_count >= CONFIRM_FRAMES:
                    current = run_team
                    out[run_start - a:i + 1] = current  # dated back to their first touch
                    run_team, run_count = NONE, 0
        out[i] = current if current != NONE else out[i]

    if current == NONE:
        current = carry_team if carry_team != NONE else HOME
        out[:] = current
    # frames before the first confirmed team (only if restart_team was NONE): already back-filled
    first = np.where(out != NONE)[0]
    if len(first):
        out[:first[0]] = out[first[0]]
    return out


def label_match(ev, dead, set_pieces, n):
    team = np.full(n, NONE, dtype=np.int8)
    # team named by the set piece that restarts an alive stretch, if any
    restart_of = {}
    for start, end, t in set_pieces:
        if t != NONE:
            restart_of[(start, end)] = t

    alive_runs = runs(~dead)
    carry = NONE
    for a, b in alive_runs:
        restart_team = NONE
        for (s, e), t in restart_of.items():
            if s - 2 * FPS <= a <= e:
                restart_team = t
                break
        seg = possession_in_stretch(ev, a, b, restart_team, carry)
        team[a:b] = seg
        carry = seg[-1]

    labels = np.full(n, "D", dtype="<U1")
    labels[team == HOME] = "H"
    labels[team == AWAY] = "A"
    labels[dead] = "D"
    return labels


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                      "rules_v1.csv"))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    pitch = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)
    meta_path, raw_path, tag_path = require(["atd_meta", "atd_raw", "smart_tagging"],
                                            GAME, args.data_dir)

    team_of_slot = read_meta(meta_path)
    print("reading tracking ({} slots) ...".format(len(team_of_slot)))
    px, py, ps, bx, by = read_atd(raw_path, len(team_of_slot), n, pitch)
    print("  ball seen on {:.1f}% of frames".format(100.0 * np.mean(~np.isnan(bx))))

    set_pieces = read_set_pieces(tag_path, n)
    print("  {} set pieces in the Smart Tagging".format(len(set_pieces)))

    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(ps, axis=1)
    activity = rolling_mean_nan(mean_speed, int(ACTIVITY_WINDOW_S * FPS))
    bspeed = ball_speed(bx, by)

    dead = dead_mask(bx, by, bspeed, activity, set_pieces, pitch)
    ev = control_evidence(px, py, team_of_slot, bx, by)
    labels = label_match(ev, dead, set_pieces, n)

    # only the two halves are graded; X elsewhere
    graded = np.zeros(n, dtype=bool)
    for _half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    labels[~graded] = "X"

    # summary
    g = labels[graded]
    n_poss = 0
    for _half, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        change = np.concatenate(([True], seg[1:] != seg[:-1]))
        n_poss += int(np.sum(change & ((seg == "H") | (seg == "A"))))
    print("graded frames: H {:.1f}%  A {:.1f}%  D {:.1f}%   possessions {}".format(
        100 * np.mean(g == "H"), 100 * np.mean(g == "A"), 100 * np.mean(g == "D"), n_poss))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
