import pytest

from src.analysis.speed import (
    MS_TO_KMH,
    SpeedEstimate,
    estimate_speed,
    instantaneous_speeds,
    peak_speed,
    peak_speeds_between_bounces,
    reject_implausible_positions,
    smooth_track,
    speeds_between_bounces,
)


def constant_velocity_track(speed_ms, fps, n_frames, start=(0.0, 0.0), axis="x"):
    """Synthetic track moving in a straight line at exactly speed_ms."""
    step = speed_ms / fps
    points = []
    for i in range(n_frames):
        if axis == "x":
            points.append((start[0] + i * step, start[1]))
        else:
            points.append((start[0], start[1] + i * step))
    return points


# --- Point estimate accuracy on synthetic constant-velocity tracks -----------------

@pytest.mark.parametrize("speed_ms,fps", [(30.0, 60), (10.0, 30), (55.0, 120), (5.0, 25)])
def test_estimate_speed_recovers_constant_velocity(speed_ms, fps):
    track = constant_velocity_track(speed_ms, fps, n_frames=fps)  # 1 second of motion

    estimate = estimate_speed(track, fps)

    assert estimate.kmh == pytest.approx(speed_ms * MS_TO_KMH, rel=1e-6)
    assert estimate.n_samples == fps


def test_estimate_speed_is_direction_independent():
    along_x = estimate_speed(constant_velocity_track(20.0, 60, 60, axis="x"), 60)
    along_y = estimate_speed(constant_velocity_track(20.0, 60, 60, axis="y"), 60)

    assert along_x.kmh == pytest.approx(along_y.kmh)


def test_estimate_speed_measures_displacement_not_path_length():
    # Ball returns to where it started: zero net displacement, so zero measured speed.
    # Documents the endpoint-displacement choice -- a caller wanting path length must
    # segment into one-directional flights first (which speeds_between_bounces does).
    track = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)]

    estimate = estimate_speed(track, fps=30)

    assert estimate.kmh == pytest.approx(0.0)


# --- Uncertainty band behaviour ----------------------------------------------------

def test_band_contains_point_estimate_and_is_asymmetric_upward():
    estimate = estimate_speed(constant_velocity_track(30.0, 60, 60), 60)

    assert estimate.kmh_low < estimate.kmh < estimate.kmh_high
    # Monocular allowance widens the top only, so the upper gap must exceed the lower.
    assert (estimate.kmh_high - estimate.kmh) > (estimate.kmh - estimate.kmh_low)


def test_longer_measurement_window_tightens_the_band():
    fps = 60
    short = estimate_speed(constant_velocity_track(30.0, fps, 12), fps)  # 0.18s
    long = estimate_speed(constant_velocity_track(30.0, fps, 60), fps)  # 0.98s

    short_width = short.kmh_high - short.kmh_low
    long_width = long.kmh_high - long.kmh_low

    assert short.kmh == pytest.approx(long.kmh)  # same true speed
    assert long_width < short_width


def test_higher_fps_alone_does_not_tighten_band_for_same_duration():
    # The band depends on the window's *duration*, not its frame count -- a real
    # constraint worth pinning: more frames over the same 0.5s doesn't beat position error.
    low_fps = estimate_speed(constant_velocity_track(30.0, 30, 15), 30)  # 0.47s
    high_fps = estimate_speed(constant_velocity_track(30.0, 120, 60), 120)  # 0.49s

    low_width = low_fps.kmh_high - low_fps.kmh_low
    high_width = high_fps.kmh_high - high_fps.kmh_low

    assert high_width == pytest.approx(low_width, rel=0.1)


def test_larger_position_uncertainty_widens_band():
    track = constant_velocity_track(30.0, 60, 60)
    tight = estimate_speed(track, 60, position_uncertainty_m=0.05)
    loose = estimate_speed(track, 60, position_uncertainty_m=0.50)

    assert (loose.kmh_high - loose.kmh_low) > (tight.kmh_high - tight.kmh_low)


def test_band_never_goes_negative():
    # Slow ball + short window: sigma exceeds the speed itself. Guards relaxed to
    # exercise the band math itself rather than the plausibility gate.
    track = [(0.0, 0.0), (0.001, 0.0)]
    estimate = estimate_speed(track, fps=30, min_samples=2, min_window_s=0.0)

    assert estimate.kmh_low == 0.0


# --- Gaps and degenerate input ------------------------------------------------------

def test_estimate_speed_tolerates_small_gaps_and_keeps_frame_timing():
    fps = 60
    full = constant_velocity_track(30.0, fps, 60)
    gapped = list(full)
    for i in range(25, 35):  # short dropout, well within the coverage floor
        gapped[i] = None

    assert estimate_speed(gapped, fps).kmh == pytest.approx(estimate_speed(full, fps).kmh)


def test_estimate_speed_rejects_span_mostly_made_of_gaps():
    # Endpoints either side of a long dropout say nothing about the motion between
    # them -- the exact window peak_speed would otherwise latch onto.
    fps = 60
    gapped = constant_velocity_track(30.0, fps, 60)
    for i in range(5, 55):
        gapped[i] = None

    assert estimate_speed(gapped, fps) is None
    assert estimate_speed(gapped, fps, min_coverage=0.0) is not None


def test_estimate_speed_returns_none_with_insufficient_data():
    assert estimate_speed([], fps=30) is None
    assert estimate_speed([(0.0, 0.0)], fps=30) is None
    assert estimate_speed([None, None, None], fps=30) is None


def test_estimate_speed_returns_none_when_window_spans_no_time():
    # Two valid points but both at the same frame index is impossible; a single
    # valid frame inside the window is the real degenerate case.
    track = [None, (1.0, 1.0), None]
    assert estimate_speed(track, fps=30) is None


def test_estimate_speed_respects_frame_range():
    fps = 60
    slow = constant_velocity_track(10.0, fps, 30)
    fast = constant_velocity_track(40.0, fps, 30, start=slow[-1])
    track = slow + fast

    slow_estimate = estimate_speed(track, fps, start_frame=0, end_frame=30)
    fast_estimate = estimate_speed(track, fps, start_frame=30, end_frame=60)

    assert slow_estimate.kmh == pytest.approx(10.0 * MS_TO_KMH, rel=1e-6)
    assert fast_estimate.kmh == pytest.approx(40.0 * MS_TO_KMH, rel=1e-6)


def test_estimate_speed_rejects_invalid_fps():
    with pytest.raises(ValueError):
        estimate_speed(constant_velocity_track(30.0, 60, 60), fps=0)


# --- Plausibility guards -------------------------------------------------------------
# These exist because real footage produced "862 km/h (839-1017)" from a 2-sample,
# 0.03s window -- correct arithmetic on bad tracking data. See docs/decisions.md.

def test_rejects_physically_impossible_speed():
    # ~360 km/h: faster than any recorded tennis shot, so it's a tracking artifact.
    track = constant_velocity_track(100.0, 60, 60)
    assert estimate_speed(track, 60) is None


def test_accepts_fast_but_plausible_serve():
    # ~230 km/h: a fast professional serve, must NOT be rejected.
    track = constant_velocity_track(64.0, 60, 60)
    estimate = estimate_speed(track, 60)

    assert estimate is not None
    assert estimate.kmh == pytest.approx(230.4, rel=1e-3)


def test_rejects_window_shorter_than_minimum():
    fps = 30
    # 3 frames at 30fps spans 0.067s -- below MIN_WINDOW_S, where one bad detection
    # swings the answer by hundreds of km/h.
    assert estimate_speed(constant_velocity_track(20.0, fps, 3), fps) is None


def test_rejects_too_few_samples():
    fps = 30
    # Long enough window, but only two valid positions inside it.
    track = [None] * 20
    track[0] = (0.0, 0.0)
    track[19] = (5.0, 0.0)

    assert estimate_speed(track, fps) is None
    # Isolate the min_samples gate from the coverage gate, which this track also trips.
    assert estimate_speed(track, fps, min_samples=2, min_coverage=0.0) is not None


def test_guards_are_configurable_for_callers_that_accept_the_risk():
    track = constant_velocity_track(100.0, 60, 60)
    assert estimate_speed(track, 60, max_speed_kmh=1000.0) is not None


def test_speeds_between_bounces_drops_implausible_segments():
    fps = 30
    # A short teleport between two bounces -- exactly the real-data failure mode.
    track = [(0.0, 0.0), (0.1, 0.0), (40.0, 0.0), (40.1, 0.0)]

    assert speeds_between_bounces(track, fps, bounce_frames=[0, 2, 3]) == []


# --- reject_implausible_positions -----------------------------------------------------

def test_reject_implausible_positions_drops_far_off_court_points():
    track = [
        (5.0, 12.0),  # mid-court, clearly valid
        (5.0, -14.1),  # 14m behind the baseline: seen in real data, a false detection
        (-30.0, 5.0),  # far off the side
        None,
    ]

    cleaned = reject_implausible_positions(track)

    assert cleaned[0] == (5.0, 12.0)
    assert cleaned[1] is None
    assert cleaned[2] is None
    assert cleaned[3] is None
    assert len(cleaned) == len(track)


def test_reject_implausible_positions_keeps_legitimately_out_balls():
    # An overhit ball landing a couple of meters past the baseline is real and must
    # survive -- Phase 2's in/out engine needs exactly these points.
    track = [(5.0, -1.5), (12.5, 10.0)]

    cleaned = reject_implausible_positions(track)

    assert cleaned == track


# --- instantaneous_speeds -----------------------------------------------------------

def test_instantaneous_speeds_on_constant_velocity():
    fps = 60
    track = constant_velocity_track(25.0, fps, 10)

    speeds = instantaneous_speeds(track, fps)

    assert speeds[0] is None  # no previous frame
    for speed in speeds[1:]:
        assert speed == pytest.approx(25.0)


def test_instantaneous_speeds_none_around_gaps():
    track = [(0.0, 0.0), None, (1.0, 0.0)]
    speeds = instantaneous_speeds(track, fps=30)
    assert speeds == [None, None, None]


# --- smooth_track --------------------------------------------------------------------

def test_smooth_track_preserves_constant_velocity():
    fps = 60
    track = constant_velocity_track(30.0, fps, 60)

    smoothed = smooth_track(track, window=5)

    # A straight-line constant-speed path is unchanged by centered averaging, except
    # at the ends where the window is clipped -- so speed over the interior is preserved.
    assert estimate_speed(smoothed, fps, start_frame=5, end_frame=55).kmh == pytest.approx(
        estimate_speed(track, fps, start_frame=5, end_frame=55).kmh, rel=1e-6
    )


def test_smooth_track_reduces_noise_spike():
    track = [(float(i), 0.0) for i in range(11)]
    track[5] = (5.0, 3.0)  # single bad detection off the line

    smoothed = smooth_track(track, window=5)

    assert smoothed[5][1] < track[5][1]


def test_smooth_track_keeps_none_gaps():
    track = [(0.0, 0.0), None, (2.0, 0.0)]
    smoothed = smooth_track(track, window=3)
    assert smoothed[1] is None
    assert len(smoothed) == len(track)


def test_smooth_track_window_below_two_is_noop():
    track = constant_velocity_track(30.0, 60, 10)
    assert smooth_track(track, window=1) == track


# --- speeds_between_bounces -----------------------------------------------------------

def test_speeds_between_bounces_segments_by_bounce_frames():
    fps = 60
    slow = constant_velocity_track(10.0, fps, 30)
    fast = constant_velocity_track(40.0, fps, 30, start=slow[-1])
    track = slow + fast

    segments = speeds_between_bounces(track, fps, bounce_frames=[0, 30, 59])

    assert len(segments) == 2
    assert segments[0][:2] == (0, 30)
    assert segments[0][2].kmh == pytest.approx(10.0 * MS_TO_KMH, rel=1e-6)
    assert segments[1][2].kmh == pytest.approx(40.0 * MS_TO_KMH, rel=1e-6)


def test_speeds_between_bounces_skips_unusable_segments():
    fps = 60
    track = constant_velocity_track(20.0, fps, 60)
    track[30:40] = [None] * 10  # this segment has no usable data

    segments = speeds_between_bounces(track, fps, bounce_frames=[0, 30, 40, 59])

    assert all(isinstance(seg[2], SpeedEstimate) for seg in segments)
    assert (30, 40) not in [seg[:2] for seg in segments]


def test_speeds_between_bounces_handles_fewer_than_two_bounces():
    track = constant_velocity_track(20.0, 60, 60)
    assert speeds_between_bounces(track, 60, bounce_frames=[]) == []
    assert speeds_between_bounces(track, 60, bounce_frames=[10]) == []


# --- peak_speed ------------------------------------------------------------------------

def test_peak_speed_matches_constant_velocity():
    fps = 60
    track = constant_velocity_track(30.0, fps, 60)

    peak = peak_speed(track, fps)

    assert peak.kmh == pytest.approx(30.0 * MS_TO_KMH, rel=1e-6)


def test_peak_speed_finds_the_fast_burst_inside_a_slow_segment():
    fps = 60
    slow_before = constant_velocity_track(3.0, fps, 30)
    burst = constant_velocity_track(35.0, fps, 30, start=slow_before[-1])
    slow_after = constant_velocity_track(3.0, fps, 30, start=burst[-1])
    track = slow_before + burst + slow_after

    peak = peak_speed(track, fps)
    whole_segment = estimate_speed(track, fps)

    assert peak.kmh == pytest.approx(35.0 * MS_TO_KMH, rel=0.05)
    # The whole-segment average is dragged far below the actual shot speed.
    assert whole_segment.kmh < peak.kmh * 0.6


def test_peak_speed_beats_endpoint_method_on_an_out_and_back_arc():
    # The real-data failure: ball leaves and returns, so net displacement ~0 even
    # though it was genuinely moving fast the whole time.
    fps = 60
    out = constant_velocity_track(25.0, fps, 30)
    back = [(out[-1][0] - (p[0] - out[0][0]), 0.0) for p in out]
    track = out + back

    assert estimate_speed(track, fps).kmh == pytest.approx(0.0, abs=1.0)
    assert peak_speed(track, fps).kmh == pytest.approx(25.0 * MS_TO_KMH, rel=0.05)


def test_peak_speed_returns_none_when_nothing_qualifies():
    assert peak_speed([None] * 60, fps=60) is None


def test_peak_speeds_between_bounces_reports_per_flight_peaks():
    fps = 60
    first = constant_velocity_track(15.0, fps, 40)
    second = constant_velocity_track(35.0, fps, 40, start=first[-1])
    track = first + second

    segments = peak_speeds_between_bounces(track, fps, bounce_frames=[0, 40, 79])

    assert len(segments) == 2
    assert segments[0][2].kmh == pytest.approx(15.0 * MS_TO_KMH, rel=0.05)
    assert segments[1][2].kmh == pytest.approx(35.0 * MS_TO_KMH, rel=0.05)
