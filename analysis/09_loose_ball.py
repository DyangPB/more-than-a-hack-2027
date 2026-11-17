"""Loose ball (tracked, but nobody within the control distance): who takes it next?

For every such frame that has a controlled frame within 3 s on both sides, compare two guesses of
the team that controls the ball next: (a) the team that controlled it last (what we do now) and
(b) the team of the nearest player. Labels are our own control rule, never the ground truth.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS  # noqa: E402
from src.control import control_from_ball, NONE  # noqa: E402

t = load_tracking()
g = graded_mask()
P, ball, team = t["players"], t["ball"], t["team"]
control, dmin = control_from_ball(ball, P, team)

d = np.hypot(P[:, :, 0] - ball[:, None, 0], P[:, :, 1] - ball[:, None, 1])
d[:, ~np.isin(team, (0, 1))] = np.inf
d = np.where(np.isnan(d), np.inf, d)
near_team = team[d.argmin(1)]

n = len(control)
pos = np.arange(n)
last = np.maximum.accumulate(np.where(control != NONE, pos, -1))
nxt = np.minimum.accumulate(np.where(control != NONE, pos, n + 10**9)[::-1])[::-1]
ok = (control == NONE) & g & np.isfinite(dmin) & (last >= 0) & (nxt < n)
ok &= (pos - last <= 3 * FPS) & (nxt - pos <= 3 * FPS)
prev_team = np.where(last >= 0, control[np.maximum(last, 0)], NONE)
next_team = control[np.minimum(nxt, n - 1)]

print("loose-ball frames with control on both sides within 3 s:", int(ok.sum()))
for lo, hi in [(2.5, 4), (4, 6), (6, 9), (9, 15), (15, 1e9)]:
    m = ok & (dmin > lo) & (dmin <= hi)
    if m.sum() < 50:
        continue
    print(f"nearest player {lo:>4}-{hi:<6} m  n={int(m.sum()):6d}  "
          f"last-team right {np.mean(prev_team[m] == next_team[m]):.3f}   "
          f"nearest-team right {np.mean(near_team[m] == next_team[m]):.3f}")
