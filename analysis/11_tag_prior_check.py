"""Hide-the-ball test for the Smart Tagging prior. Tracking + Smart Tagging only, never the ground truth.

Hide the ball on blocks where it IS tracked, decode, and compare with what our control rule saw on the
hidden frames (an algorithmic pseudo-label). The players-only model and the tag reliabilities are both
fitted on the other 5-minute blocks only. Also prints the number of possession runs on the real data, to
check that the extra evidence does not change the granularity set by the 1 s assumption.
"""
import os
import sys
import warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS, GRADED  # noqa: E402
from src.control import control_from_ball, NONE, HOME  # noqa: E402
from src.deadball import (median_player_speed, dead_from_set_pieces,  # noqa: E402
                          dead_from_still_players, dead_before_kickoff)
from src.playform import team_shape_features, fit_logistic, logit_from_model  # noqa: E402
from src.decode import viterbi_two_state, evidence, SWITCH_COST  # noqa: E402
from src.tagprior import tag_logit, BLOCK_FRAMES  # noqa: E402
import load_smart_tagging as st  # noqa: E402

warnings.simplefilter("ignore")
t = load_tracking()
g = graded_mask()
control, _ = control_from_ball(t["ball"], t["players"], t["team"])
events = st.load()
speed = median_player_speed(t["players"])
dsp, _ = dead_from_set_pieces(events, speed)
live = g & ~((dsp | dead_from_still_players(speed) | dead_before_kickoff()) & g)
N = len(control)
ball_missing = np.isnan(t["ball"][:, 0])

feats = team_shape_features(t["players"], t["team"])
blk = np.arange(N) // BLOCK_FRAMES
yH = (control == HOME).astype(float)
oof = np.zeros(N)
for a, b in GRADED:
    idx = np.arange(a, b + 1)
    lab = idx[(control[idx] != NONE) & ~np.isnan(feats[idx]).any(1)]
    for bid in np.unique(blk[idx]):
        te, tr = idx[blk[idx] == bid], lab[blk[lab] != bid]
        oof[te] = logit_from_model(fit_logistic(feats[tr], yH[tr]), feats[te])
tag_oof = tag_logit(events, control, g, held_out=True)

edge = np.diff(np.concatenate([[0], live.astype(np.int8), [0]]))
stretches = list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def decode(ctrl, hid, wp, wt, cost=SWITCH_COST):
    out = np.full(N, NONE, dtype=np.int8)
    for s, e in stretches:
        d = evidence(ctrl[s:e], oof[s:e], wp, (ball_missing | hid)[s:e], None, 0.0, tag_oof[s:e], wt)
        out[s:e] = viterbi_two_state(d, cost)
    return out


def hidden(seed):
    rng = np.random.default_rng(seed)
    h = np.zeros(N, dtype=bool)
    for a, b in GRADED:
        i = a + int(rng.integers(0, 25 * FPS))
        while i < b:
            i += int(rng.integers(10, 40) * FPS)
            j = i + int(rng.integers(10, 60) * FPS)
            h[i:min(j, b)] = True
            i = j
    return h


combos = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.02), (0.0, 0.05), (0.0, 0.1), (0.0, 0.2),
          (0.1, 0.02), (0.1, 0.05), (0.1, 0.1), (0.1, 0.2)]
acc = {c: [] for c in combos}
for seed in range(4):
    h = hidden(seed)
    c2 = control.copy()
    c2[h] = NONE
    ev = h & live & (control != NONE)
    for wp, wt in combos:
        acc[(wp, wt)].append((decode(c2, h, wp, wt)[ev] == control[ev]).mean())
print("hidden-ball frames evaluated:", int(ev.sum()))
print("players weight  tag weight   accuracy   possession runs on the real data")
no_hide = np.zeros(N, dtype=bool)
for wp, wt in combos:
    o = decode(control, no_hide, wp, wt)
    r = 1 + int((o[live][1:] != o[live][:-1]).sum())
    print(f"{wp:14.2f}  {wt:10.2f}   {np.mean(acc[(wp, wt)]):.3f}      {r}")
