"""Per-shot ball speed estimation from a court-coordinate trajectory.

Consumes ball positions already projected into court meters (see
src/perception/homography.py's project_track) plus the video fps, and returns
speeds in km/h **with an uncertainty band** -- never a bare number. The band is the
point of this module: at 30 fps a single frame pair spans only 33 ms, so a ~15 cm
position error already implies a ~20 km/h speed error. Reporting "142 km/h" from
that data would be false precision.

Two known error sources, both documented in docs/decisions.md:

* **Random** -- ball-detection centroid + homography error, modelled as
  `position_uncertainty_m`. Shrinks with a longer measurement window, so speed is
  measured endpoint-to-endpoint across a shot rather than frame-to-frame.
* **Systematic** -- a monocular camera only recovers the ball's *ground-plane*
  motion, so a ball travelling with any vertical component moves further in 3D than
  we measure. The true speed is therefore at or above the measured value, which is
  why the returned band is asymmetric (see MONOCULAR_UNDERESTIMATE_ALLOWANCE).
"""

import math
from dataclasses import dataclass
from itertools import pairwise

MS_TO_KMH = 3.6

# Combined ball-detection centroid + homography reprojection error, in meters. Rough
# but honest: keypoints land within a few cm on good footage, ball centroid is the
# bigger contributor on a blurred fast ball.
DEFAULT_POSITION_UNCERTAINTY_M = 0.15

# A ball with vertical motion covers more 3D distance than its ground-plane shadow,
# so the measured (2D) speed is a floor on the true speed. Widens the band upward only.
MONOCULAR_UNDERESTIMATE_ALLOWANCE = 0.15

# --- Plausibility guards -------------------------------------------------------------
# Without these, bad inputs produce confident nonsense: on real footage a 2-sample,
# 0.03s window yielded "862 km/h (839-1017)" -- the uncertainty math was correct, the
# ball positions feeding it were not. Speed is displacement over a short time, so it
# amplifies tracking errors; refusing to answer beats answering wrongly.

# Fastest serve ever recorded is ~263 km/h. Anything above this is a tracking artifact,
# not a shot.
MAX_PLAUSIBLE_SPEED_KMH = 260.0

# Below this window the random position error dominates (see the sigma term) and a
# single mis-detected frame swings the answer by hundreds of km/h.
MIN_WINDOW_S = 0.10

MIN_SAMPLES = 3

# Fraction of frames within the measured span that must actually have a ball position.
# A window whose endpoints sit either side of a tracking gap says nothing about what
# the ball did in between, and `peak_speed` -- being a maximum -- actively seeks those
# windows out. On footage with ~50% ball detection this alone produced 250+ km/h
# "shots" in amateur rallies.
MIN_COVERAGE = 0.5

# How far outside the court a ball position can plausibly land. Generous: an overhit
# ball legitimately lands several meters past the baseline, and an airborne ball
# projects further still through a ground-plane homography (see docs/decisions.md).
# Beyond this it's a false detection on the crowd/fence/background.
COURT_PLAUSIBILITY_MARGIN_M = 6.0


@dataclass(frozen=True)
class SpeedEstimate:
    """A speed measurement and its uncertainty. Always treat kmh as an estimate."""

    kmh: float  # point estimate, from ground-plane displacement
    kmh_low: float
    kmh_high: float
    duration_s: float  # measurement window actually used
    n_samples: int  # valid positions that contributed

    def __str__(self):
        return f"{self.kmh:.0f} km/h ({self.kmh_low:.0f}-{self.kmh_high:.0f})"


def reject_implausible_positions(points_m, margin_m=COURT_PLAUSIBILITY_MARGIN_M):
    """None-out ball positions too far outside the court to be real.

    Ball tracking occasionally locks onto background objects (crowd, fence, a line
    judge), projecting to positions like 14 m behind the baseline. Those points are
    indistinguishable from real ones downstream and wreck any speed computed across
    them, so drop them before measuring.

    Deliberately generous: this removes obvious false detections, not the subtler
    error where a genuinely airborne ball projects past its true ground position.
    """
    from src.scoring import rules

    cleaned = []
    for point in points_m:
        if point is None:
            cleaned.append(None)
            continue
        x, y = point
        in_x = -margin_m <= x <= rules.DOUBLES_WIDTH_M + margin_m
        in_y = -margin_m <= y <= rules.DOUBLES_LENGTH_M + margin_m
        cleaned.append(point if (in_x and in_y) else None)
    return cleaned


def smooth_track(points_m, window=5):
    """Moving-average smooth a court-coordinate track, preserving None gaps.

    Averages each point with its valid neighbours inside `window` (centered). Frames
    that are None stay None -- gaps are not interpolated here, since inventing ball
    positions would silently corrupt both speed and in/out calls.
    """
    if window < 2:
        return list(points_m)

    half = window // 2
    smoothed = []
    for i, point in enumerate(points_m):
        if point is None:
            smoothed.append(None)
            continue
        neighbours = [
            p for p in points_m[max(0, i - half): i + half + 1] if p is not None
        ]
        xs = sum(p[0] for p in neighbours) / len(neighbours)
        ys = sum(p[1] for p in neighbours) / len(neighbours)
        smoothed.append((xs, ys))
    return smoothed


def instantaneous_speeds(points_m, fps):
    """Frame-to-frame speed in m/s, index-aligned with points_m.

    Entry i is the speed between frame i-1 and i, or None when either frame lacks a
    position. Noisy by nature (see the module docstring) -- useful for finding the
    peak of a shot, not for reporting a number.
    """
    speeds = [None]
    for i in range(1, len(points_m)):
        previous, current = points_m[i - 1], points_m[i]
        if previous is None or current is None:
            speeds.append(None)
            continue
        distance = math.dist(previous, current)
        speeds.append(distance * fps)
    return speeds


def estimate_speed(
    points_m,
    fps,
    start_frame=0,
    end_frame=None,
    position_uncertainty_m=DEFAULT_POSITION_UNCERTAINTY_M,
    min_samples=MIN_SAMPLES,
    min_window_s=MIN_WINDOW_S,
    max_speed_kmh=MAX_PLAUSIBLE_SPEED_KMH,
    min_coverage=MIN_COVERAGE,
):
    """Estimate speed over a frame range from endpoint displacement.

    Uses the first and last valid positions in the range rather than averaging
    frame-to-frame speeds: over a window of duration T the random position error
    contributes only sigma*sqrt(2)/T, so a longer window is strictly better, whereas
    per-frame differencing amplifies noise at every step.

    Returns None rather than a number whenever the data can't support one -- too few
    samples, too short a window, or a physically impossible result. Callers should
    treat None as "unknown", not zero.

    :param points_m: per-frame (x, y) court-meters positions, or None
    :param fps: video frame rate
    :param start_frame: inclusive start of the measurement window
    :param end_frame: exclusive end; defaults to the end of points_m
    :param min_samples: minimum valid positions required
    :param min_window_s: minimum measurement duration required
    :param max_speed_kmh: results above this are treated as tracking artifacts
    :param min_coverage: fraction of frames in the measured span needing a position
    :return: SpeedEstimate, or None
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    end_frame = len(points_m) if end_frame is None else min(end_frame, len(points_m))
    start_frame = max(0, start_frame)

    valid = [
        (i, points_m[i])
        for i in range(start_frame, end_frame)
        if points_m[i] is not None
    ]
    if len(valid) < min_samples:
        return None

    (first_index, first_point), (last_index, last_point) = valid[0], valid[-1]
    duration_s = (last_index - first_index) / fps
    if duration_s < min_window_s:
        return None

    spanned_frames = last_index - first_index + 1
    if len(valid) / spanned_frames < min_coverage:
        return None

    speed_ms = math.dist(first_point, last_point) / duration_s
    if speed_ms * MS_TO_KMH > max_speed_kmh:
        return None

    # Two independent endpoint errors -> sqrt(2); a longer window divides it down.
    sigma_ms = position_uncertainty_m * math.sqrt(2) / duration_s

    kmh = speed_ms * MS_TO_KMH
    kmh_low = max(0.0, (speed_ms - sigma_ms) * MS_TO_KMH)
    kmh_high = (speed_ms + sigma_ms) * MS_TO_KMH * (1 + MONOCULAR_UNDERESTIMATE_ALLOWANCE)

    return SpeedEstimate(
        kmh=kmh,
        kmh_low=kmh_low,
        kmh_high=kmh_high,
        duration_s=duration_s,
        n_samples=len(valid),
    )


def peak_speed(points_m, fps, window_s=0.25, start_frame=0, end_frame=None, **kwargs):
    """Fastest sustained speed over any window_s-long window in the range.

    Preferred over a whole-segment `estimate_speed` for "how hard was that hit?".
    Endpoint displacement across a full bounce-to-bounce interval measures *net*
    displacement, which collapses toward zero whenever the ball arcs back to near
    where it started or the segment includes dead time between rallies -- on real
    footage that produced 0 km/h for genuine shots. A short sliding window instead
    catches the ball's actual flight.

    window_s trades noise against dilution: too short and position error dominates,
    too long and the peak is averaged away with slower parts of the flight.

    :return: the best SpeedEstimate found, or None if no window qualifies
    """
    end_frame = len(points_m) if end_frame is None else min(end_frame, len(points_m))
    start_frame = max(0, start_frame)
    window_frames = max(2, round(window_s * fps))

    best = None
    for window_start in range(start_frame, end_frame - window_frames + 1):
        estimate = estimate_speed(
            points_m, fps,
            start_frame=window_start,
            end_frame=window_start + window_frames + 1,
            **kwargs,
        )
        if estimate is not None and (best is None or estimate.kmh > best.kmh):
            best = estimate
    return best


def peak_speeds_between_bounces(points_m, fps, bounce_frames, **kwargs):
    """Per-flight peak speed -- the per-shot number to actually report.

    Same segmentation as speeds_between_bounces, but each segment is measured by its
    peak window rather than endpoint displacement.
    """
    boundaries = sorted(set(bounce_frames))
    segments = []
    for start, end in pairwise(boundaries):
        estimate = peak_speed(points_m, fps, start_frame=start, end_frame=end, **kwargs)
        if estimate is not None:
            segments.append((start, end, estimate))
    return segments


def speeds_between_bounces(points_m, fps, bounce_frames, **kwargs):
    """Estimate a speed for each flight segment between consecutive bounces.

    A shot's flight is the span between two bounces, which is the closest proxy for
    "one shot" available from cached perception data alone -- true contact frames
    need racket/player context that Phase 4's rally segmentation will add.

    :return: list of (start_frame, end_frame, SpeedEstimate); segments with too
        little data are skipped
    """
    boundaries = sorted(set(bounce_frames))
    segments = []
    for start, end in pairwise(boundaries):
        estimate = estimate_speed(points_m, fps, start_frame=start, end_frame=end, **kwargs)
        if estimate is not None:
            segments.append((start, end, estimate))
    return segments
