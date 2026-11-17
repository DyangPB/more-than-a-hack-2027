"""Extra dead-ball layers from the teammate DyangPB's own pipeline (DyangPB/mtah-DeculGirl-submission,
src/deculgirl/features/dead_ball.py), re-implemented on this repository's arrays for v4.

Five tracking-only layers; a frame is dead if ANY of them says so:

  out      the ball is detected outside the pitch for at least 1 s
  still    the ball stays inside a 1 m circle for 1 s AND the players are at walking pace or slower
  quiet    the players' median speed stays under 0.5 m/s for 10 s (injury, substitution, VAR; the ball is
           often not tracked there)
  huddle   the players are clustered (RMS spread from the centroid < 16 m) AND slower than normal play
           for 5 s (goal celebration, substitution, gathering around the referee)
  prekick  every frame from the start of the second graded half until its kick off

Every number is a football or physics assumption (the median speed of the tracked players is about
1.35 m/s through the graded halves, so thresholds on it are low), none is fitted to the match, to its
public ground truth or to its video. Results are cleaned like the original: gaps of up to 1 s inside a
dead stretch are bridged, dead stretches shorter than 2 s are dropped, and the pre-kick-off block stays.
"""
import warnings

import numpy as np
import pandas as pd

from src.tracking import FPS, GRADED, graded_mask  # noqa: E402
from games import GAMES, GAME  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

PITCH = GAMES[GAME]["pitch"]
WALK_SPEED = 1.0
STILL_RADIUS_M = 1.0
STILL_FRAMES = FPS
OUT_FRAMES = FPS
QUIET_SPEED = 0.5
QUIET_FRAMES = 10 * FPS
MIN_DEAD_FRAMES = 2 * FPS
BRIDGE_FRAMES = FPS
SPEED_SPAN = FPS
HUDDLE_SPREAD_M = 16.0
HUDDLE_FRAMES = 5 * FPS
MIN_TRACKED = 14
MAX_PLAYER_SPEED = 12.0


def team_speed(players):
    """Median speed (m/s) over tracked slots, displacement across SPEED_SPAN frames, centred."""
    half = SPEED_SPAN // 2
    n = players.shape[0]
    xy = players[:, :, :2]
    ahead = np.full_like(xy, np.nan)
    behind = np.full_like(xy, np.nan)
    ahead[: n - half] = xy[half:]
    behind[half:] = xy[: n - half]
    disp = np.hypot(*(ahead - behind).transpose(2, 0, 1))
    speed = disp * FPS / (2 * half)
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(speed, axis=1)


def team_spread(players):
    """RMS distance (m) of the tracked players from their centroid; NaN with fewer than MIN_TRACKED."""
    xy = players[:, :, :2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cx = np.nanmean(xy[:, :, 0], axis=1)
        cy = np.nanmean(xy[:, :, 1], axis=1)
        rms = np.sqrt(np.nanmean((xy[:, :, 0] - cx[:, None]) ** 2 + (xy[:, :, 1] - cy[:, None]) ** 2, axis=1))
    rms[np.isfinite(xy[:, :, 0]).sum(axis=1) < MIN_TRACKED] = np.nan
    return rms


def _runs(mask):
    edge = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))


def _close(mask, gap):
    """Fill False gaps of at most `gap` frames between True stretches."""
    out = mask.copy()
    prev_end = None
    for s, e in _runs(mask):
        if prev_end is not None and s - prev_end <= gap:
            out[prev_end:s] = True
        prev_end = e
    return out


def _keep_long(mask, min_len):
    out = np.zeros_like(mask)
    for s, e in _runs(mask):
        if e - s >= min_len:
            out[s:e] = True
    return out


def dead_extra(ball, players, speed=None):
    """Dead-ball mask from the five tracking-only layers, on graded frames."""
    n = len(ball)
    bx, by = ball[:, 0], ball[:, 1]
    det = np.isfinite(bx)
    if speed is None:
        speed = team_speed(players)
    slow = np.nan_to_num(speed, nan=np.inf) < WALK_SPEED
    quiet = np.nan_to_num(speed, nan=np.inf) < QUIET_SPEED

    out_of_pitch = det & ((bx < 0) | (bx > PITCH[0]) | (by < 0) | (by > PITCH[1]))
    out = _keep_long(_close(out_of_pitch, BRIDGE_FRAMES), OUT_FRAMES)

    w = STILL_FRAMES
    sx, sy = pd.Series(bx), pd.Series(by)
    rx = sx.rolling(w, center=True, min_periods=w // 2 + 1).max() - sx.rolling(
        w, center=True, min_periods=w // 2 + 1).min()
    ry = sy.rolling(w, center=True, min_periods=w // 2 + 1).max() - sy.rolling(
        w, center=True, min_periods=w // 2 + 1).min()
    still = det & (np.hypot(rx.to_numpy(), ry.to_numpy()) <= STILL_RADIUS_M) & slow

    quiet_long = _keep_long(_close(quiet, BRIDGE_FRAMES), QUIET_FRAMES)

    clustered = np.nan_to_num(team_spread(players), nan=np.inf) < HUDDLE_SPREAD_M
    huddle = _keep_long(_close(clustered & slow, BRIDGE_FRAMES), HUDDLE_FRAMES)

    prekick = np.zeros(n, dtype=bool)
    for (start, _end), half in zip(GRADED, ("1st Half", "2nd Half")):
        prekick[start:KICKOFF_VIDEO_FRAMES[half]] = True

    g = graded_mask()
    dead = out | still | quiet_long | huddle | prekick
    dead = _keep_long(_close(dead & g, BRIDGE_FRAMES), MIN_DEAD_FRAMES)
    dead |= prekick
    return dead & g
