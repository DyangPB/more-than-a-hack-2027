"""Layer 2c: decode one stretch of live play into HOME / AWAY with an explicit switching cost.

Replaces "hold the last team": every frame gives some evidence for HOME or AWAY, and the team only
changes when the accumulated evidence for the other side outweighs the cost of a change.

    evidence d_t > 0 favours HOME, < 0 favours AWAY (units: frames of clear ball control)
      - a player of a team controls the ball:      +1 / -1
      - nobody controls it and the ball is untracked: WEIGHT * logit(players-only model)
      - ball tracked, nobody controls it, nearest player 2.5-4 m away: +/- SOFT_WEIGHT for that
        player's team (analysis/09_loose_ball.py: at that distance the nearest team takes the
        ball next 75% of the time, against 65% for "the team that had it last"; beyond 6 m it is
        worse than holding, so nothing is added there)
      - a Smart Tagging phase covers the frame and nobody controls the ball: TAG_WEIGHT * tag log-odds
        (src/tagprior.py)
      - otherwise                                      0
    switching cost: SWITCH_COST units

SWITCH_COST = 15 is calibrated to the earlier assumption (a possession change needs about 1 s of
control), not to any score: because opposing touches cancel evidence here, a cost of 25 would
merge possessions that the 25-frame confirmation rule keeps apart (161 runs against 233 over the
live frames). At 15 and weight 0 the decoder gives 231 runs, i.e. the same granularity as that
rule, and only then is the players-only evidence added on top.
"""
import numpy as np

HOME, AWAY = 0, 1
NONE_SOFT = -1
SWITCH_COST = 15.0


def viterbi_two_state(d, switch_cost=SWITCH_COST):
    """Best HOME/AWAY path for per-frame evidence d, paying switch_cost per change.

    State score: H gets +d/2, A gets -d/2 each frame. Returns an int8 array of HOME/AWAY.
    """
    n = len(d)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    half = np.asarray(d, dtype=float) / 2.0
    back = np.zeros((n, 2), dtype=bool)          # True: this state was entered by a switch
    sh, sa = half[0], -half[0]
    for i in range(1, n):
        stay_h, swi_h = sh, sa - switch_cost
        stay_a, swi_a = sa, sh - switch_cost
        back[i, 0] = swi_h > stay_h
        back[i, 1] = swi_a > stay_a
        sh = max(stay_h, swi_h) + half[i]
        sa = max(stay_a, swi_a) + (-half[i])
    path = np.zeros(n, dtype=np.int8)
    s = 0 if sh >= sa else 1
    for i in range(n - 1, -1, -1):
        path[i] = s
        if back[i, s]:
            s = 1 - s
    return path


def evidence(control, players_logit, weight, ball_missing=None, soft=None, soft_weight=0.0,
             tag=None, tag_weight=0.0):
    """Per-frame evidence d_t from control observations and the players-only model.

    The players-only model is only used where the ball is untracked (ball_missing True), which is
    the situation it was built and tested for. ball_missing=None means everywhere.
    """
    d = np.zeros(len(control))
    d[control == HOME] = 1.0
    d[control == AWAY] = -1.0
    free = (control != HOME) & (control != AWAY)
    if ball_missing is not None:
        free &= ball_missing
    d[free] = weight * players_logit[free]
    if soft is not None:
        near = (soft != NONE_SOFT) & (control != HOME) & (control != AWAY)
        d[near] += soft_weight * np.where(soft[near] == HOME, 1.0, -1.0)
    if tag is not None and tag_weight:
        nobody = (control != HOME) & (control != AWAY)
        d[nobody] += tag_weight * tag[nobody]
    return d
