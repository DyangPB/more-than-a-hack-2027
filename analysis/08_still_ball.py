"""Dead-ball candidate: the ball itself standing still. Only tracking + Smart Tagging, never the ground truth.

A live ball is almost always moving (passes, dribbles, clearances). A ball that stays put for
several seconds is waiting for a restart. This prints how long such stretches are, how much of
them the existing dead-ball layers already cover, and whether a tracker that freezes the last
ball position could be mistaken for one.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS  # noqa: E402
from src.deadball import (median_player_speed, dead_from_set_pieces,  # noqa: E402
                          dead_from_still_players, dead_before_kickoff)
import load_smart_tagging as st  # noqa: E402

t = load_tracking()
g = graded_mask()
ball = t["ball"]
vis = ~np.isnan(ball[:, 0])

# exact repeats of the ball coordinates = a frozen tracker, not a physical ball
rep = np.zeros(len(ball), dtype=bool)
rep[1:] = vis[1:] & vis[:-1] & (ball[1:, 0] == ball[:-1, 0]) & (ball[1:, 1] == ball[:-1, 1])
print("visible graded frames:", int((vis & g).sum()), " exact coordinate repeats:", int((rep & g).sum()))

# ball speed over 1 s (m/s), NaN where the ball is not tracked at both ends
k = FPS // 2
sp = np.full(len(ball), np.nan)
d = ball[2 * k:] - ball[:-2 * k]
sp[k:-k] = np.hypot(d[:, 0], d[:, 1]) / (2 * k / FPS)
print("ball speed quantiles (m/s) on graded visible frames:",
      np.round(np.nanquantile(sp[g], [.05, .1, .25, .5, .75, .9]), 2))

speed = median_player_speed(t["players"])
dsp, _ = dead_from_set_pieces(st.load(), speed)
existing = (dsp | dead_from_still_players(speed) | dead_before_kickoff()) & g
print("existing dead share of graded frames: %.3f" % (existing.sum() / g.sum()))

for thr in (0.3, 0.5, 1.0):
    for secs in (3, 5, 8):
        still = np.nan_to_num(sp, nan=np.inf) < thr
        edge = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
        m = np.zeros(len(ball), dtype=bool)
        for s, e in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
            if e - s >= secs * FPS:
                m[s:e] = True
        m &= g
        new = m & ~existing
        print(f"ball < {thr} m/s for >= {secs} s: {m.sum() / g.sum():.3f} of graded, "
              f"new beyond existing {new.sum() / g.sum():.3f}, runs {int(np.sum(np.diff(m.astype(int)) == 1))}")
