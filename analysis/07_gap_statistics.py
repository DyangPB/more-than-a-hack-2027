"""Descriptive only: how long is the ball missing, and how often does the team change across a gap?
Uses algorithmic control labels (nearest player within 2.5 m), never the ground truth."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import numpy as np
from src.tracking import load_tracking, graded_mask, FPS
from src.control import control_from_ball, NONE

t = load_tracking()
g = graded_mask()
control, dmin = control_from_ball(t["ball"], t["players"], t["team"])
ball_ok = ~np.isnan(t["ball"][:, 0])
print("graded frames", g.sum(), "ball visible", (ball_ok & g).sum() / g.sum())
print("control observed", ((control != NONE) & g).sum() / g.sum())

idx = np.flatnonzero((control != NONE) & g)
gaps = np.diff(idx)
same = control[idx[1:]] == control[idx[:-1]]
for lo, hi in [(1, 2), (2, 13), (13, 51), (51, 126), (126, 251), (251, 10**7)]:
    m = (gaps >= lo) & (gaps < hi)
    print(f"gap {lo/FPS:5.1f}-{hi/FPS:7.1f}s  n={m.sum():5d}  team changes across gap: {(~same[m]).mean():.3f}")
print("share of graded frames inside gaps > 5 s between control observations:",
      gaps[gaps > 125].sum() / g.sum())
