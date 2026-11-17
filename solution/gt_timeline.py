"""Print the public 5-minute ground truth as a readable timeline, to watch next to the video.

    python solution/gt_timeline.py

For understanding the possession definition only (allowed by the rules: "inspecting the provided
data and the video to better understand what counts as possession"). It shows the ground truth,
never our predictions, and nothing in the pipeline reads its output.

Video clock = frame / 25, the time shown by a player on the tactical video.
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def clock(frame):
    s = frame / 25.0
    return "{:02d}:{:05.2f}".format(int(s // 60), s % 60)


def main():
    labels = pd.read_csv(GT)["label"].astype(str).str.strip().values
    names = {"H": "H  Barcelona", "A": "A  Athletic ", "D": "D  dead ball"}
    start = None
    print("{:>3}  {:<13} {:>9} -> {:<9} {:>6}   frames".format("#", "label", "from", "to", "dur s"))
    k = 0
    for i in range(len(labels) + 1):
        lab = labels[i] if i < len(labels) else None
        if start is not None and lab != labels[start]:
            if labels[start] != "X":
                k += 1
                print("{:>3}  {:<13} {:>9} -> {:<9} {:6.1f}   {}-{}".format(
                    k, names[labels[start]], clock(start), clock(i - 1), (i - start) / 25.0,
                    start, i - 1))
            start = i
        elif start is None:
            start = i


if __name__ == "__main__":
    main()
