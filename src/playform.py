"""Layer 2b: which team has the ball when the ball itself is not tracked -- from the players only.

The ball is untracked on ~41% of graded frames, often for tens of seconds, and over such a gap the
team changes about as often as it stays (see analysis/07_gap_statistics.py). Holding the last team
is therefore close to a coin flip there. The shape of the two teams does carry some signal (the
team in possession is further up the pitch, more spread, moving forward), so we learn it.

Self-supervised, no human labels: the training labels are the team that *our own control rule*
(control.py) sees controlling the ball on frames where the ball IS tracked. The model only ever
sees player positions, never the ball, so it is learning exactly the missing-ball situation. It is
a low-capacity logistic regression (9 inputs) fitted separately per half, because the teams swap
ends.

Nothing here reads the ground truth, and nothing is tuned on it.
"""
import warnings
import numpy as np
import pandas as pd

from src.tracking import FPS, GRADED  # noqa: E402
from src.control import HOME, AWAY, NONE  # noqa: E402

PITCH_MID_X = 52.5          # m, the halfway line of a 105 m pitch
SMOOTH_FRAMES = FPS         # 1 s, same smoothing as the speed signal
L2 = 1.0                    # ridge strength on standardised inputs


def _team_stats(players, mask):
    """Per frame: centroid x/y, spread x/y, mean x-velocity and mean speed of one team."""
    x = players[:, mask, :]
    k = FPS // 2
    v = np.full_like(x, np.nan)
    v[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2 * k / FPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (np.nanmean(x[..., 0], 1), np.nanmean(x[..., 1], 1),
                np.nanstd(x[..., 0], 1), np.nanstd(x[..., 1], 1),
                np.nanmean(v[..., 0], 1), np.nanmean(np.hypot(v[..., 0], v[..., 1]), 1))


def team_shape_features(players, team):
    """(N, 9) float features from the two teams' shapes. NaN where a team is not tracked."""
    h = _team_stats(players, team == HOME)
    a = _team_stats(players, team == AWAY)
    cols = {"d_cx": h[0] - a[0], "d_cy": h[1] - a[1], "d_sx": h[2] - a[2], "d_sy": h[3] - a[3],
            "d_vx": h[4] - a[4], "d_sp": h[5] - a[5], "sum_sp": h[5] + a[5]}
    mid = (h[0] + a[0]) / 2 - PITCH_MID_X          # where the play is, along the pitch
    cols["play_x"] = mid
    cols["d_cx_x_play"] = (h[0] - a[0]) * mid      # who is ahead, weighted by where the play is
    f = pd.DataFrame(cols).rolling(SMOOTH_FRAMES, min_periods=1, center=True).mean()
    return f.to_numpy()


def fit_logistic(x, y, l2=L2, iters=100):
    """Ridge logistic regression by Newton steps. Deterministic. Returns (mean, std, weights)."""
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.c_[(x - mu) / sd, np.ones(len(x))]
    w = np.zeros(z.shape[1])
    pen = np.r_[np.ones(z.shape[1] - 1), 0.0] * l2
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        grad = z.T @ (p - y) + pen * w
        hess = (z * (p * (1 - p) + 1e-6)[:, None]).T @ z + np.diag(pen + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return mu, sd, w


def logit_from_model(model, x):
    """log-odds that HOME has the ball, per row of x (0 where any input is NaN)."""
    mu, sd, w = model
    ok = ~np.isnan(x).any(axis=1)
    out = np.zeros(len(x))
    out[ok] = np.c_[(x[ok] - mu) / sd, np.ones(ok.sum())] @ w
    return out


def players_only_logit(players, team, control, graded):
    """Per frame log-odds (HOME vs AWAY) from the players' shape, one model per half.

    Fitted on the frames of that half where the control rule sees a team with the ball.
    """
    feats = team_shape_features(players, team)
    out = np.zeros(len(players))
    y_all = (control == HOME).astype(float)
    for a, b in GRADED:
        sl = slice(a, b + 1)
        x = feats[sl]
        use = (control[sl] != NONE) & graded[sl] & ~np.isnan(x).any(axis=1)
        if use.sum() < 1000:
            continue
        model = fit_logistic(x[use], y_all[sl][use])
        out[sl] = logit_from_model(model, x)
    return out
