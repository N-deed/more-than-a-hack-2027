"""Calibrate the rules on EXTERNAL data: Metrica Sports open Sample Game 3.

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../metrica-sample-data
    python solution/calibrate_metrica.py --sample ../metrica-sample-data/data/Sample_Game_3

Nothing here touches the challenge match. Sample Game 3 is a different, anonymised game that
Metrica publishes with synchronised tracking (EPTS format, the same as the challenge ATD) and
event data. We turn its events into a possession reference and run our pipeline on its tracking.

Reference built from the events (our own derivation, documented in README.md):
  - dead ball: from the end of the last event before each SET PIECE to the set piece itself,
    plus the time before each half's kick-off;
  - otherwise: the team of the most recent on-ball event (PASS, CARRY, RECOVERY, SET PIECE,
    SHOT, BALL LOST). CHALLENGE events are ignored: both sides emit them.

The Sample Game 3 tracking is cleaner than the challenge ATD (ball visible far more often), so
we degrade it to match the challenge conditions before running the rules: the ball is removed
on ~41% of in-play frames in gaps of random length, players drop out at random, and positions
get ~0.3 m of noise. The seed is fixed, so the calibration is reproducible.
"""

import argparse
import itertools
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession as P  # noqa: E402
import possession_metrics as M  # noqa: E402

PITCH = (105.0, 68.0)
FPS = 25
ON_BALL = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}


def load_tracking(folder):
    meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    teams = [t.get("id") for t in meta.iter("Team")]
    home_id = teams[0]
    team_of_player = {p.get("id"): (P.HOME if p.get("teamId") == home_id else P.AWAY)
                      for p in meta.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        slots = [team_of_player[channel_player[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))

    lines = open(os.path.join(folder, "Sample_Game_3_tracking.txt")).read().split("\n")
    lines = [l for l in lines if l]
    n = len(lines)
    S = max(len(s[2]) for s in specs)
    px = np.full((n, S), np.nan)
    py = np.full((n, S), np.nan)
    team = np.full((n, S), P.NONE, dtype=np.int8)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    for line in lines:
        frame_s, players, ball = line.split(":")
        i = int(frame_s) - 1
        for k, g in enumerate(players.split(";")):
            if g[0] != "N":
                x, y = g.split(",")[:2]
                px[i, k] = float(x) * PITCH[0]
                py[i, k] = float(y) * PITCH[1]
        if ball[0] != "N":
            x, y = ball.split(",")[:2]
            bx[i] = float(x) * PITCH[0]
            by[i] = float(y) * PITCH[1]
    for start, end, slots in specs:
        team[start - 1:end, :len(slots)] = slots
    return px, py, team, bx, by, specs


def reference(folder, n, specs):
    events = json.load(open(os.path.join(folder, "Sample_Game_3_events.json")))["data"]
    events.sort(key=lambda e: e["start"]["frame"])
    meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    home_id = [t.get("id") for t in meta.iter("Team")][0]
    second_start = specs[1][0]
    halves = [(0, second_start - 1), (second_start - 1, n)]

    labels = np.full(n, "X", dtype="<U1")
    for h0, h1 in halves:
        evs = [e for e in events if h0 <= e["start"]["frame"] < h1]
        if not evs:
            continue
        first = evs[0]["start"]["frame"]
        last = max(e["end"]["frame"] or e["start"]["frame"] for e in evs)
        current = None
        team_at = np.full(h1 - h0, "D", dtype="<U1")
        idx = 0
        for t in range(first, last):
            while idx < len(evs) and evs[idx]["start"]["frame"] <= t:
                e = evs[idx]
                if e["type"]["name"] in ON_BALL and e.get("team"):
                    current = "H" if e["team"]["id"] == home_id else "A"
                idx += 1
            team_at[t - h0] = current or "D"
        prev_end = None
        for e in evs:
            if e["type"]["name"] == "SET PIECE" and prev_end is not None:
                team_at[prev_end - h0:e["start"]["frame"] - h0] = "D"
            prev_end = e["end"]["frame"] or e["start"]["frame"]
        labels[first:last] = team_at[first - h0:last - h0]
    return labels


def degrade(px, py, bx, by, seed=0, missing=0.41, noise=0.3):
    rng = np.random.default_rng(seed)
    n = len(bx)
    bx, by = bx.copy(), by.copy()
    px = px + rng.normal(0, noise, px.shape)
    py = py + rng.normal(0, noise, py.shape)
    bx = bx + rng.normal(0, noise, n)
    by = by + rng.normal(0, noise, n)
    drop = np.isnan(bx)
    while drop.mean() < missing:
        start = rng.integers(0, n)
        length = int(rng.exponential(2.0 * FPS)) + 1
        drop[start:start + length] = True
    bx[drop] = np.nan
    by[drop] = np.nan
    pdrop = rng.random(px.shape) < 0.05
    px[pdrop] = np.nan
    py[pdrop] = np.nan
    return px, py, bx, by


def player_speed(px, py, lag=5):
    """Per-player speed (m/s) over `lag` frames, as a tracking provider would report it."""
    v = np.hypot(px[lag:] - px[:-lag], py[lag:] - py[:-lag]) * FPS / lag
    v[v > 12] = np.nan
    return np.vstack([np.full((lag, px.shape[1]), np.nan), v])


def run_rules(px, py, team, bx, by, ps):
    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(ps, axis=1)
    activity = P.rolling_mean_nan(mean_speed, int(P.ACTIVITY_WINDOW_S * FPS))
    bspeed = P.ball_speed(bx, by)
    dead = P.dead_mask(bx, by, bspeed, activity, [], PITCH)
    ev = P.control_evidence(px, py, team, bx, by)
    return P.label_match(ev, dead, [], len(bx), ~np.isnan(bx))


def score(gt, pred):
    mask = gt == "X"
    pred = pred.copy()
    pred[mask] = "X"
    r = M.evaluate(M.label_codes(gt), M.label_codes(pred), bootstrap_n=0)
    return r


def summary(r):
    def find(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    args = ap.parse_args()
    px, py, team, bx, by, specs = load_tracking(args.sample)
    gt = reference(args.sample, len(bx), specs)
    print("reference: H {} A {} D {} X {}".format(*(int(np.sum(gt == c)) for c in "HADX")))
    dpx, dpy, dbx, dby = degrade(px, py, bx, by)
    ps = player_speed(px, py)
    pred = run_rules(dpx, dpy, team, dbx, dby, ps)
    r = score(gt, pred)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, (dict, list))}, indent=1,
                     default=str)[:3000])
