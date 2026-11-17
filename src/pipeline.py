"""Assemble the layers into one label per video frame: H / A / D / X.

    python -m src.pipeline [name]      -> evaluation/results/<name>.csv   (run from the repo root)
"""
import os
import sys
import numpy as np
import pandas as pd

from src.tracking import load_tracking, graded_mask, FPS, GRADED, N_FRAMES, ROOT  # noqa: E402
from src.control import control_from_ball, confirm_control, soft_control, HOME, AWAY, NONE  # noqa: E402
from src.deadball import (median_player_speed, dead_from_set_pieces,  # noqa: E402
                          dead_from_still_players, dead_before_kickoff)
from src.decode import viterbi_two_state, evidence, SWITCH_COST  # noqa: E402
from src.playform import players_only_logit  # noqa: E402
from src.tagprior import tag_logit  # noqa: E402
from src.deadball_extra import dead_extra  # noqa: E402
import load_smart_tagging as st  # noqa: E402

# x4: weight of the players-only evidence relative to one frame of clear ball control. In a
# self-supervised test (hide the ball where it is tracked, decode, compare with what the control
# rule saw) accuracy rises from 0.63 at weight 0 to 0.73 at 0.1 and stays flat up to 0.5; we take
# the smallest value on that plateau so the team only changes on sustained evidence.
PLAYERS_WEIGHT = 0.1
# x5: weight of "nearest team of a loose ball, 2.5-4 m away" relative to one frame of control.
# About half: that guess is right ~75% of the time (logit 1.1) against ~90% for real control (2.2).
SOFT_WEIGHT = 0.5
# x6: x4 with the switching cost set so that, with the players-only weight at 0, the decoder
# segments the live frames exactly like v3 (18-frame confirmation): 307 runs at cost 10, v3 gives 307.
V6_SWITCH_COST = 10.0
# x7: x4 plus the Smart Tagging phase prior. In the hide-the-ball test (analysis/11_tag_prior_check.py,
# reliabilities fitted on the other 5-minute blocks) accuracy goes 0.732 -> 0.743 from weight 0.05 up to 0.1;
# we take the smallest value on that plateau.
TAG_WEIGHT = 0.05
# x8: x6 (v3's granularity) plus the tag prior.


def team_series(control, live, min_control_frames=25):
    """Team in possession (HOME/AWAY/NONE) per frame.

    Computed separately inside every stretch of live play, so possession never carries over a
    stoppage: after a restart the first team that controls the ball owns the frames before it.
    Live stretches where nobody is ever seen controlling the ball borrow the nearest known team.
    """
    out = np.full(len(control), NONE, dtype=np.int8)
    edge = np.diff(np.concatenate([[0], live.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        seg = confirm_control(control[s:e], min_frames=min_control_frames)
        seen = np.flatnonzero(seg != NONE)
        if len(seen):
            seg[:seen[0]] = seg[seen[0]]          # look-ahead: restart belongs to who takes the ball
        out[s:e] = seg
    known = np.flatnonzero(out != NONE)
    todo = np.flatnonzero((out == NONE) & live)
    if len(known) and len(todo):
        pos = np.searchsorted(known, todo)
        out[todo] = out[known[np.maximum(pos - 1, 0)]]
    return out


def team_series_decoded(control, live, players_logit, ball_missing, soft=None, weight=PLAYERS_WEIGHT,
                        soft_weight=0.0, cost=SWITCH_COST, tag=None, tag_weight=0.0):
    """Team in possession per frame, x4: a switching-cost decoder inside every live stretch.

    Stretches stay independent (possession never carries over a stoppage), as in team_series.
    """
    out = np.full(len(control), NONE, dtype=np.int8)
    edge = np.diff(np.concatenate([[0], live.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        d = evidence(control[s:e], players_logit[s:e], weight, ball_missing[s:e],
                     None if soft is None else soft[s:e], soft_weight,
                     None if tag is None else tag[s:e], tag_weight)
        out[s:e] = viterbi_two_state(d, cost)
    return out


def build_labels(version="v2"):
    """Build v2 (reference), v3 (marked experiment), v4 (switching-cost decoder + players-only evidence)
    v5 (v4 + weak evidence from the nearest team of a loose ball) or v6 (v4 at v3's granularity)."""
    t = load_tracking()
    graded = graded_mask()
    # === V3 IMPLEMENTATION: responsive possession confirmation ===
    if version in ("v3", "v4"):
        # Slightly shorter confirmation improves responsiveness while retaining hysteresis.
        control, _ = control_from_ball(t["ball"], t["players"], t["team"], max_dist=2.5)
        min_control = 18
    else:
        control, _ = control_from_ball(t["ball"], t["players"], t["team"])
        min_control = 25
    speed = median_player_speed(t["players"])
    dead_sp, _ = dead_from_set_pieces(st.load(), speed)
    if version == "v4":
        # v4: the dead ball is DyangPB's tracking layers plus the Smart Tagging set-piece windows
        # (the still-players rule of v2/v3 is not used, its job is done by the quiet layer).
        dead = (dead_sp | dead_extra(t["ball"], t["players"])) & graded
    else:
        dead = (dead_sp | dead_from_still_players(speed) | dead_before_kickoff()) & graded
    live = graded & ~dead
    if version in ("x4", "x5", "x6", "x7", "x8"):
        logit = players_only_logit(t["players"], t["team"], control, graded)
        soft = soft_control(t["ball"], t["players"], t["team"]) if version == "x5" else None
        team = team_series_decoded(control, live, logit, np.isnan(t["ball"][:, 0]), soft,
                                   soft_weight=SOFT_WEIGHT if version == "x5" else 0.0,
                                   cost=V6_SWITCH_COST if version in ("x6", "x8") else SWITCH_COST,
                                   tag=tag_logit(st.load(), control, graded) if version in ("x7", "x8") else None,
                                   tag_weight=TAG_WEIGHT if version in ("x7", "x8") else 0.0)
    else:
        team = team_series(control, live, min_control_frames=min_control)
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    labels[live & (team == HOME)] = "H"
    labels[live & (team == AWAY)] = "A"
    labels[dead] = "D"
    return labels


def main(name="v1"):
    labels = build_labels(name)
    path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(path, index=False)
    print("wrote", path, "({} rows)".format(len(labels)))
    g = graded_mask()
    print("graded frames:", {k: int((labels[g] == k).sum()) for k in "HADX"})
    for half, (s, e) in enumerate(GRADED, 1):
        seg = labels[s:e + 1]
        runs = 1 + int((seg[1:] != seg[:-1]).sum())
        print("half {}: {} label runs".format(half, runs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
