"""What does the Smart Tagging `Team` tag mean? Check it against OUR control rule, never the ground truth.

For every phase code: how often the tagged team is the team our rule sees controlling the ball while the
phase is running (frames where the ball is tracked and a player is within 2.5 m), and how much of the
frames where the ball is NOT tracked the phase windows cover.
"""
import os
import sys
import collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask  # noqa: E402
from src.control import control_from_ball, NONE, HOME, AWAY  # noqa: E402
import load_smart_tagging as st  # noqa: E402

t = load_tracking()
g = graded_mask()
control, _ = control_from_ball(t["ball"], t["players"], t["team"])
missing = np.isnan(t["ball"][:, 0])
events = st.load()

names = collections.Counter(e["tags"].get("Team", "") for e in events)
print("Team tag values:", dict(names))
to_team = {"Barcelona": HOME, "Atletic Club": AWAY}

print(f"{'code':32s} {'n':>4s} {'frames w/ control':>18s} {'tag == controller':>18s}")
tot_cov = np.zeros(len(control), dtype=bool)
for code in sorted({e["code"] for e in events}):
    agree = total = 0
    for e in events:
        if e["code"] != code or e["tags"].get("Team") not in to_team:
            continue
        a, b = e["start_frame"], e["end_frame"]
        c = control[a:b]
        ok = (c != NONE) & g[a:b]
        total += int(ok.sum())
        agree += int((c[ok] == to_team[e["tags"]["Team"]]).sum())
        tot_cov[a:b] = True
    n = sum(1 for e in events if e["code"] == code)
    if total:
        print(f"{code:32s} {n:4d} {total:18d} {agree / total:18.3f}")
print("graded frames covered by any phase window with a team tag: %.3f" % ((tot_cov & g).sum() / g.sum()))
print("... of the frames where the ball is not tracked:            %.3f" % ((tot_cov & g & missing).sum() / (g & missing).sum()))
