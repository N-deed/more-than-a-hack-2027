"""Load the three Metrica Sports open sample games (EXTERNAL data) into one common shape.

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../metrica-sample-data

Games 1 and 2 are in Metrica's CSV format, game 3 in EPTS (the format of the challenge ATD).
Every loader returns the same dict:

    px, py   (n, S) player positions in metres, NaN when missing
    team     (n, S) HOME / AWAY / NONE per player column
    bx, by   (n,)   ball position in metres
    period   (n,)   1 or 2, 0 outside play
    ref      (n,)   possession reference derived from the events: H / A / D / X

The reference is our own derivation from Metrica's published events (see README.md section 3):
dead from the end of the last event before each set piece to that set piece; otherwise the team
of the most recent on-ball event (PASS, CARRY, RECOVERY, SET PIECE, SHOT, BALL LOST).
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HOME, AWAY, NONE = 0, 1, -1
PITCH = (105.0, 68.0)
FPS = 25
ON_BALL = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}


def _reference(events, n):
    """events: list of dicts {team: 'H'/'A'/None, type, start, end, period}, frames 0-based."""
    ref = np.full(n, "X", dtype="<U1")
    for period in (1, 2):
        evs = sorted([e for e in events if e["period"] == period], key=lambda e: e["start"])
        if not evs:
            continue
        first = evs[0]["start"]
        last = max(max(e["end"], e["start"]) for e in evs)
        lab = np.full(last - first, "D", dtype="<U1")
        current, k = None, 0
        for t in range(first, last):
            while k < len(evs) and evs[k]["start"] <= t:
                if evs[k]["type"] in ON_BALL and evs[k]["team"]:
                    current = evs[k]["team"]
                k += 1
            lab[t - first] = current or "D"
        prev_end = None
        for e in evs:
            if e["type"] == "SET PIECE" and prev_end is not None and e["start"] > prev_end:
                lab[prev_end - first:e["start"] - first] = "D"
            prev_end = max(e["end"], e["start"])
        ref[first:last] = lab
    return ref


def _period_array(n, events):
    period = np.zeros(n, dtype=np.int8)
    for p in (1, 2):
        evs = [e for e in events if e["period"] == p]
        if evs:
            a = min(e["start"] for e in evs)
            b = max(max(e["end"], e["start"]) for e in evs)
            period[a:b] = p
    return period


def load_csv_game(folder, k):
    stem = os.path.join(folder, "Sample_Game_{}".format(k), "Sample_Game_{}_".format(k))
    sides = []
    for side, code in (("Home", HOME), ("Away", AWAY)):
        df = pd.read_csv(stem + "RawTrackingData_{}_Team.csv".format(side), header=2)
        vals = df.values[:, 3:].astype(float)
        cols = list(df.columns[3:])
        ball_col = cols.index("Ball")
        players = vals[:, :ball_col]
        ball = vals[:, ball_col:ball_col + 2]
        sides.append((players[:, 0::2], players[:, 1::2], code, ball))
    n = min(s[0].shape[0] for s in sides)
    px = np.hstack([s[0][:n] for s in sides]) * PITCH[0]
    py = np.hstack([s[1][:n] for s in sides]) * PITCH[1]
    team = np.hstack([np.full((n, s[0].shape[1]), s[2], dtype=np.int8) for s in sides])
    bx = sides[0][3][:n, 0] * PITCH[0]
    by = sides[0][3][:n, 1] * PITCH[1]

    ev = pd.read_csv(stem + "RawEventsData.csv")
    events = []
    for _, r in ev.iterrows():
        t = r["Team"]
        events.append({"team": "H" if t == "Home" else ("A" if t == "Away" else None),
                       "type": str(r["Type"]), "start": int(r["Start Frame"]) - 1,
                       "end": int(r["End Frame"]) - 1, "period": int(r["Period"])})
    return {"px": px, "py": py, "team": team, "bx": bx, "by": by,
            "period": _period_array(n, events), "ref": _reference(events, n),
            "name": "game{}".format(k)}


def load_epts_game(folder):
    g = os.path.join(folder, "Sample_Game_3")
    meta = ET.parse(os.path.join(g, "Sample_Game_3_metadata.xml")).getroot()
    home_id = [t.get("id") for t in meta.iter("Team")][0]
    team_of_player = {p.get("id"): (HOME if p.get("teamId") == home_id else AWAY)
                      for p in meta.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        slots = [team_of_player[channel_player[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))
    lines = [l for l in open(os.path.join(g, "Sample_Game_3_tracking.txt")).read().split("\n") if l]
    n = len(lines)
    S = max(len(s[2]) for s in specs)
    px = np.full((n, S), np.nan)
    py = np.full((n, S), np.nan)
    team = np.full((n, S), NONE, dtype=np.int8)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    for line in lines:
        f, players, ball = line.split(":")
        i = int(f) - 1
        for j, grp in enumerate(players.split(";")):
            if grp[0] != "N":
                x, y = grp.split(",")[:2]
                px[i, j] = float(x) * PITCH[0]
                py[i, j] = float(y) * PITCH[1]
        if ball[0] != "N":
            x, y = ball.split(",")[:2]
            bx[i] = float(x) * PITCH[0]
            by[i] = float(y) * PITCH[1]
    for a, b, slots in specs:
        team[a - 1:b, :len(slots)] = slots
    raw = json.load(open(os.path.join(g, "Sample_Game_3_events.json")))["data"]
    events = []
    for e in raw:
        tid = (e.get("team") or {}).get("id")
        events.append({"team": None if tid is None else ("H" if tid == home_id else "A"),
                       "type": e["type"]["name"], "start": int(e["start"]["frame"]),
                       "end": int(e["end"]["frame"] or e["start"]["frame"]),
                       "period": int(e["period"])})
    return {"px": px, "py": py, "team": team, "bx": bx, "by": by,
            "period": _period_array(n, events), "ref": _reference(events, n), "name": "game3"}


def load_all(folder):
    return [load_csv_game(folder, 1), load_csv_game(folder, 2), load_epts_game(folder)]
