"""Who has the ball, judged from the shape of the two teams alone (no ball needed).

A team in possession pushes up, opens the pitch and moves forward; the team without the ball
drops, narrows and moves back. This module turns that into a probability, P(home in possession),
from a few collective features, with a logistic regression trained on EXTERNAL data only
(Metrica open sample games, see train_shape_model.py). The fitted coefficients are stored in
shape_model.json, so running the pipeline needs numpy only.

Features (all computed per team in its own attacking direction, then home minus away, so the
model is symmetric: swapping the teams flips the probability):
    advance   mean position along the pitch (how high the block is)
    vel       mean speed along the pitch towards the opponent goal (1 s window)
    width     spread across the pitch (std of y)
    length    spread along the pitch (std of x)
    line      height of the last outfield line (second deepest player)
"""

import json
import os
import warnings

import numpy as np

HOME, AWAY, NONE = 0, 1, -1
FPS = 25
HALF_LENGTH = 52.5
FEATURES = ["advance", "vel", "width", "length", "line"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_model.json")


def _rolling_mean(values, window):
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


def attack_direction(px, team, period):
    """{(period, team): +1 if the team attacks towards x = 105, -1 otherwise}."""
    out = {}
    for p in np.unique(period[period > 0]):
        m = period == p
        means = {}
        for t in (HOME, AWAY):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means[t] = np.nanmean(np.where(team[m] == t, px[m], np.nan))
        out[(p, HOME)] = 1.0 if means[HOME] < means[AWAY] else -1.0
        out[(p, AWAY)] = -out[(p, HOME)]
    return out


def team_features(px, py, team, period, lag=FPS, smooth=FPS):
    """(n, len(FEATURES)) array, home minus away, NaN where a team is not visible."""
    n, S = px.shape
    if team.ndim == 1:
        team = np.broadcast_to(team[None, :], (n, S))
    direction = attack_direction(px, team, period)
    vx = np.full_like(px, np.nan)
    vx[lag:] = (px[lag:] - px[:-lag]) * FPS / lag
    vx[np.abs(vx) > 12] = np.nan                     # identity switches, not running

    per_team = {}
    for t in (HOME, AWAY):
        d = np.zeros(n)
        for p in (1, 2):
            d[period == p] = direction.get((p, t), 0.0)
        m = team == t
        x = np.where(m, px, np.nan)
        y = np.where(m, py, np.nan)
        v = np.where(m, vx, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rel = d[:, None] * (x - HALF_LENGTH)       # positive = towards the opponent goal
            advance = np.nanmean(rel, axis=1)
            vel = d * np.nanmean(v, axis=1)
            width = np.nanstd(y, axis=1)
            length = np.nanstd(x, axis=1)
            srt = np.sort(np.where(np.isnan(rel), np.inf, rel), axis=1)
            line = srt[:, 1]
            line[~np.isfinite(line)] = np.nan
        per_team[t] = np.vstack([advance, vel, width, length, line]).T
    diff = per_team[HOME] - per_team[AWAY]
    diff[(period == 0)] = np.nan
    for j in range(diff.shape[1]):
        diff[:, j] = _rolling_mean(diff[:, j], smooth)
    return diff


def load_model(path=MODEL_PATH):
    with open(path) as handle:
        m = json.load(handle)
    return np.array(m["mean"]), np.array(m["scale"]), np.array(m["coef"])


def predict_home(features, model=None):
    """P(home in possession) per frame, 0.5 where the features are missing."""
    mean, scale, coef = model if model is not None else load_model()
    z = (features - mean) / scale
    logit = np.nan_to_num(z) @ coef
    p = 1.0 / (1.0 + np.exp(-logit))
    p[np.isnan(features).any(axis=1)] = 0.5
    return p


def fit_logistic(X, y, l2=1.0, iters=50):
    """Logistic regression without intercept (the model is symmetric), Newton's method."""
    mean = np.zeros(X.shape[1])                       # no centring: keeps the symmetry
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = X / scale
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        g = Z.T @ (p - y) + l2 * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mean, scale, w
