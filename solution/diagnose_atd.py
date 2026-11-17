"""Describe the ball-tracking gaps of the challenge ATD. Uses the tracking only, no ground truth.

    python solution/diagnose_atd.py

Prints the length distribution of the gaps where the ball is not tracked, inside the two halves,
and how the players move during short and long gaps. These numbers set the gap model used to
degrade the external calibration game (solution/calibrate_metrica.py) so that it resembles the
challenge tracking.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import possession as P  # noqa: E402


def main():
    pitch = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    team = P.read_meta(meta_path)
    px, py, ps, bx, by = P.read_atd(raw_path, len(team), n, pitch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        activity = P.rolling_mean_nan(np.nanmean(ps, axis=1), 75)

    graded = np.zeros(n, dtype=bool)
    for _h, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    seen = ~np.isnan(bx)
    gaps = [(a, b) for a, b in P.runs(~seen & graded)]
    lengths = np.array([b - a for a, b in gaps]) / 25.0
    vis = np.array([b - a for a, b in P.runs(seen & graded)]) / 25.0
    missing = np.sum(~seen & graded)

    print("graded frames {}  ball missing {:.1f}%".format(graded.sum(),
                                                         100.0 * missing / graded.sum()))
    print("gaps {}  visible runs {}".format(len(lengths), len(vis)))
    print("gap length (s)      p25 {:.2f}  p50 {:.2f}  p75 {:.2f}  p90 {:.2f}  p99 {:.1f}".format(
        *np.percentile(lengths, [25, 50, 75, 90, 99])))
    print("visible run (s)     p25 {:.2f}  p50 {:.2f}  p75 {:.2f}  p90 {:.2f}".format(
        *np.percentile(vis, [25, 50, 75, 90])))
    edges = [0, 0.2, 0.5, 1, 2, 4, 8, 15, 30, 1e9]
    print("share of missing frames by gap length:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lengths >= lo) & (lengths < hi)
        frames = np.sum(lengths[m]) * 25
        act = [np.nanmean(activity[a:b]) for (a, b), k in zip(gaps, m) if k]
        print("  {:>5}-{:<5}s  gaps {:>5}  missing frames {:5.1f}%  mean player speed {:.2f}".format(
            lo, hi if hi < 1e8 else "inf", int(m.sum()), 100.0 * frames / missing,
            np.nanmean(act) if act else float("nan")))
    print("player speed when ball visible: {:.2f}".format(np.nanmean(activity[seen & graded])))


if __name__ == "__main__":
    main()
