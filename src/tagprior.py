"""Layer 2d: Smart Tagging phases as a prior on who has the ball.

Every phase (BUILD UP, PROGRESSION, DEFENDING IN MIDDLE THIRD, ...) carries a Team tag, and the tag is
about the team the phase describes, not always the team with the ball: for the DEFENDING phases the
controller is the other team (analysis/10_team_tags.py: tag == controller 0.16-0.21 there, 0.58-0.84 in
the attacking phases). Rather than assume the direction of each code, we measure it against our own
control rule on frames where the ball is tracked (an algorithmic pseudo-label, never the ground truth)
and turn each code into a signed log-odds:

    reliability(code) = logit( max(agree, 1 - agree) ),   sign = +1 if agree > 0.5 else -1

The frame's tag evidence is the sum over the phase windows that cover it, as log-odds of HOME.
"""
import numpy as np

from src.tracking import N_FRAMES  # noqa: E402
from src.control import HOME, AWAY, NONE  # noqa: E402

TEAM_OF_TAG = {"Barcelona": HOME, "Atletic Club": AWAY}
MIN_FRAMES = 500            # a code with less control evidence than this gets no weight
BLOCK_FRAMES = 5 * 60 * 25  # 5 minutes, for the held-out validation


def _windows(events):
    """(code, team, start, end) for every phase with a known team."""
    out = []
    for e in events:
        team = TEAM_OF_TAG.get(e["tags"].get("Team"))
        if team is not None:
            out.append((e["code"], team, e["start_frame"], min(e["end_frame"], N_FRAMES)))
    return out


def _counts(windows, control, graded, n_blocks):
    """agree / total frames per code and per 5-minute block."""
    codes = sorted({w[0] for w in windows})
    agree = {c: np.zeros(n_blocks) for c in codes}
    total = {c: np.zeros(n_blocks) for c in codes}
    for code, team, a, b in windows:
        for blk in range(a // BLOCK_FRAMES, (b - 1) // BLOCK_FRAMES + 1):
            lo, hi = max(a, blk * BLOCK_FRAMES), min(b, (blk + 1) * BLOCK_FRAMES)
            c = control[lo:hi]
            ok = (c != NONE) & graded[lo:hi]
            total[code][blk] += ok.sum()
            agree[code][blk] += (c[ok] == team).sum()
    return agree, total


def _signed_logit(agree, total):
    if total < MIN_FRAMES:
        return 0.0
    p = min(max(agree / total, 1e-3), 1 - 1e-3)
    return float(np.sign(p - 0.5) * np.log(max(p, 1 - p) / min(p, 1 - p)))


def tag_logit(events, control, graded, held_out=False):
    """Per-frame log-odds that HOME has the ball, from the phase windows (0 where none covers a frame).

    held_out=True: in every 5-minute block, each code's reliability comes from the other blocks only.
    """
    n_blocks = N_FRAMES // BLOCK_FRAMES + 1
    windows = _windows(events)
    agree, total = _counts(windows, control, graded, n_blocks)
    out = np.zeros(N_FRAMES)
    for code, team, a, b in windows:
        for blk in range(a // BLOCK_FRAMES, (b - 1) // BLOCK_FRAMES + 1):
            lo, hi = max(a, blk * BLOCK_FRAMES), min(b, (blk + 1) * BLOCK_FRAMES)
            ag, tot = agree[code].sum(), total[code].sum()
            if held_out:
                ag, tot = ag - agree[code][blk], tot - total[code][blk]
            s = _signed_logit(ag, tot)
            out[lo:hi] += s if team == HOME else -s
    return out
