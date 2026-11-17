"""Load the SkillCorner open data (EXTERNAL, 20 A-League 2024/25 matches) into the common shape.

    git clone https://github.com/SkillCorner/opendata.git ../skillcorner-opendata   (git lfs)
    # or download data/matches/<id>/<id>_tracking_extrapolated.jsonl, _match.json,
    # _phases_of_play.csv for every match into one folder as <id>.jsonl, <id>_match.json,
    # <id>_phases.csv

Broadcast tracking at 10 fps, resampled here to 25 fps by linear interpolation so that every
threshold of possession.py keeps its meaning. Only ball positions SkillCorner marks as detected
are kept; extrapolated ones are dropped, like a tracker that has lost the ball.

Reference (our derivation, see README.md section 3): SkillCorner's phases of play give the team
in possession over time. A gap between two phases shorter than PHASE_GAP_S is a loose ball and
keeps the previous team (the challenge rule); a longer gap is a stoppage (D). Frames where the
broadcast shows no usable tracking (replays, close-ups: fewer than 10 players) are not scored.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

HOME, AWAY, NONE = 0, 1, -1
SRC_FPS, FPS = 10, 25
PHASE_GAP_S = 3.0


def _resample(arr10, n25):
    """Linear interpolation of a (n10, k) array onto the 25 fps grid; NaN if a neighbour is NaN."""
    t = np.arange(n25) * SRC_FPS / FPS
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(arr10) - 1)
    w = (t - i0)[:, None]
    return arr10[i0] * (1 - w) + arr10[i1] * w


def load_match(folder, match_id):
    meta = json.load(open(os.path.join(folder, "{}_match.json".format(match_id))))
    home_id = meta["home_team"]["id"]
    length, width = meta["pitch_length"], meta["pitch_width"]
    col = {}
    team_cols = []
    for p in meta["players"]:
        col[p["id"]] = len(col)
        team_cols.append(HOME if p["team_id"] == home_id else AWAY)
    frames = [json.loads(l) for l in open(os.path.join(folder, "{}.jsonl".format(match_id)))]
    n10 = len(frames)
    S = len(col)
    px = np.full((n10, S), np.nan)
    py = np.full((n10, S), np.nan)
    ball = np.full((n10, 2), np.nan)
    period10 = np.zeros(n10, dtype=np.int8)
    nplayers = np.zeros(n10)
    for i, f in enumerate(frames):
        period10[i] = f["period"] or 0
        b = f["ball_data"]
        if b and b.get("is_detected") and b["x"] is not None:
            ball[i] = (b["x"] + length / 2.0, b["y"] + width / 2.0)
        for p in f["player_data"]:
            j = col.get(p["player_id"])
            if j is not None and p["x"] is not None:
                px[i, j] = p["x"] + length / 2.0
                py[i, j] = p["y"] + width / 2.0
        nplayers[i] = sum(1 for p in f["player_data"] if p.get("is_detected"))

    # reference at 10 fps
    ref10 = np.full(n10, "X", dtype="<U1")
    ph = pd.read_csv(os.path.join(folder, "{}_phases.csv".format(match_id)))
    ph = ph.sort_values("frame_start")
    for period in (1, 2):
        rows = ph[ph.period == period]
        prev = None
        for _, r in rows.iterrows():
            lab = "H" if r.team_in_possession_id == home_id else "A"
            a, b = int(r.frame_start), int(r.frame_end)
            if prev is not None:
                gap_a, gap_b, prev_lab = prev
                if a > gap_a:
                    fill = prev_lab if (a - gap_a) < PHASE_GAP_S * SRC_FPS else "D"
                    ref10[gap_a:a] = fill
            ref10[a:b] = lab
            prev = (b, b, lab)
    ref10[nplayers < 10] = "X"

    n25 = int((n10 - 1) * FPS / SRC_FPS) + 1
    px25 = _resample(px, n25)
    py25 = _resample(py, n25)
    ball25 = _resample(ball, n25)
    src = np.round(np.arange(n25) * SRC_FPS / FPS).astype(int).clip(0, n10 - 1)
    return {"px": px25, "py": py25, "team": np.array(team_cols, dtype=np.int8),
            "bx": ball25[:, 0], "by": ball25[:, 1], "period": period10[src],
            "ref": ref10[src], "pitch": (float(length), float(width)),
            "name": "sc{}".format(match_id)}


def load_all(folder):
    ids = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(folder, "*.jsonl")))
    return [load_match(folder, i) for i in ids]
