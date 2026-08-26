import pytest

from src.perception.court_detection import CourtDetectionResult
from src.pipeline import STAGE_WEIGHTS, build_analysis, cache_dir_for
from src.scoring import rules

FPS = 30.0


class FakeCourt(CourtDetectionResult):
    """A CourtDetectionResult with a fixed, exact pixel->meters homography.

    Uses a plain scale so a court position in meters is trivially (px/100, py/100) --
    keeping these tests about the assembly logic rather than homography accuracy,
    which tests/test_homography.py already covers.
    """

    def __init__(self, n_frames):
        matrix = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]
        super().__init__(
            keypoints_px=[[(0.0, 0.0)] * 14] * n_frames,
            homography_m=[matrix] * n_frames,
        )


def px(point_m):
    """Inverse of FakeCourt's homography: meters -> pixels."""
    return (point_m[0] * 100, point_m[1] * 100)


NEAR_IN = (rules.CENTER_X_M, rules.NET_Y_M + 4.0)
NEAR_OUT = (rules.CENTER_X_M, rules.DOUBLES_LENGTH_M + 2.0)


def make_inputs(n_frames, positions, bounce_frames):
    """positions: {frame: point_in_meters}."""
    ball = [None] * n_frames
    for frame, point in positions.items():
        ball[frame] = px(point)
    return {
        "video": "clip.mp4",
        "fps": FPS,
        "n_frames": n_frames,
        "ball_track": ball,
        "court_result": FakeCourt(n_frames),
        "bounce_frames": bounce_frames,
    }


# --- Result shape ---------------------------------------------------------------------

def test_build_analysis_reports_detection_coverage():
    inputs = make_inputs(100, {i: NEAR_IN for i in range(0, 100, 2)}, bounce_frames=[])

    result = build_analysis(**inputs)

    assert result.video == "clip.mp4"
    assert result.n_frames == 100
    assert result.ball_detected_frames == 50
    assert result.court_detected_frames == 100
    assert len(result.ball_track_m) == 100


def test_result_is_json_serializable():
    import json

    inputs = make_inputs(60, {30: NEAR_OUT}, bounce_frames=[30])

    payload = build_analysis(**inputs).to_dict()

    assert json.loads(json.dumps(payload))["bounces"][0]["frame"] == 30


# --- Bounce calls ---------------------------------------------------------------------

def test_bounces_carry_in_out_calls():
    inputs = make_inputs(60, {20: NEAR_IN, 40: NEAR_OUT}, bounce_frames=[20, 40])

    result = build_analysis(**inputs)

    assert [b.frame for b in result.bounces] == [20, 40]
    assert result.bounces[0].is_in is True
    assert result.bounces[1].is_in is False
    assert result.bounces[1].distance_m < 0


def test_bounce_without_a_nearby_position_is_dropped():
    # Frame 50 has no tracked position within the search window.
    inputs = make_inputs(60, {20: NEAR_IN}, bounce_frames=[20, 50])

    result = build_analysis(**inputs)

    assert [b.frame for b in result.bounces] == [20]


def test_bounce_borrows_a_position_from_a_nearby_frame():
    inputs = make_inputs(60, {21: NEAR_OUT}, bounce_frames=[20])

    result = build_analysis(**inputs)

    assert len(result.bounces) == 1
    assert result.bounces[0].frame == 20


def test_court_type_changes_the_call():
    alley = (rules.LEFT_SINGLES_X_M / 2, rules.NET_Y_M + 4.0)
    inputs = make_inputs(60, {30: alley}, bounce_frames=[30])

    singles = build_analysis(**inputs, court_type="singles")
    doubles = build_analysis(**inputs, court_type="doubles")

    assert singles.bounces[0].is_in is False
    assert doubles.bounces[0].is_in is True


def test_implausible_positions_are_excluded():
    way_off_court = (rules.CENTER_X_M, -40.0)
    inputs = make_inputs(60, {30: way_off_court}, bounce_frames=[30])

    result = build_analysis(**inputs)

    assert result.bounces == []
    assert result.ball_track_m[30] is None


# --- Speeds and rallies -----------------------------------------------------------------

def test_shots_are_reported_between_bounces():
    # A steady 15 m/s traverse between two bounces 40 frames apart.
    positions = {}
    for i in range(41):
        positions[i] = (rules.CENTER_X_M, rules.NET_Y_M - 6.0 + i * (15.0 / FPS))
    inputs = make_inputs(60, positions, bounce_frames=[0, 40])

    result = build_analysis(**inputs)

    assert len(result.shots) == 1
    shot = result.shots[0]
    assert shot.start_frame == 0
    assert shot.kmh == pytest.approx(15.0 * 3.6, rel=0.1)
    assert shot.kmh_low < shot.kmh < shot.kmh_high


def test_rallies_are_segmented_and_carry_a_proposal():
    positions = {10: NEAR_IN, 40: NEAR_OUT, 400: NEAR_OUT}
    inputs = make_inputs(500, positions, bounce_frames=[10, 40, 400])

    result = build_analysis(**inputs)

    assert len(result.rallies) == 2, "a 12s gap should split the rallies"
    first = result.rallies[0]
    assert first.bounce_frames == [10, 40]
    assert first.proposed_winner == 0
    assert 0 < first.proposal_confidence <= 0.6


def test_no_bounces_yields_empty_analysis_not_an_error():
    inputs = make_inputs(60, {10: NEAR_IN}, bounce_frames=[])

    result = build_analysis(**inputs)

    assert result.bounces == []
    assert result.shots == []
    assert result.rallies == []


# --- Progress reporting -------------------------------------------------------------------

def test_progress_fractions_are_monotonic_and_bounded():
    from src.pipeline import _progress_reporter

    seen = []
    report = _progress_reporter(lambda fraction, stage, message: seen.append(fraction))

    for stage in STAGE_WEIGHTS:
        report(stage, 0.0)
        report(stage, 1.0)

    assert seen == sorted(seen)
    assert seen[0] == pytest.approx(0.0)
    assert seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_progress_is_optional():
    from src.pipeline import _progress_reporter

    report = _progress_reporter(None)
    report("court", 0.5)  # must not raise


def test_stage_weights_sum_to_one():
    assert sum(STAGE_WEIGHTS.values()) == pytest.approx(1.0)


# --- Intra-stage progress ----------------------------------------------------------
# The vendored inference loops take tens of seconds with no callback; without this a
# progress bar sits frozen. Both access patterns they use must report.

def test_progress_frames_reports_while_being_iterated():
    from src.pipeline import _ProgressFrames

    seen = []
    frames = _ProgressFrames(list(range(100)), seen.append)

    consumed = list(frames)

    assert consumed == list(range(100)), "iteration must still yield the real frames"
    assert len(seen) > 5
    assert seen == sorted(seen)
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_progress_frames_reports_while_being_indexed():
    from src.pipeline import _ProgressFrames

    seen = []
    frames = _ProgressFrames(list(range(100)), seen.append)

    for i in range(100):
        assert frames[i] == i

    assert len(seen) > 5
    assert seen == sorted(seen)


def test_progress_never_goes_backwards_when_the_model_looks_back():
    """ball_detector reads frames[num], frames[num-1], frames[num-2]."""
    from src.pipeline import _ProgressFrames

    seen = []
    frames = _ProgressFrames(list(range(100)), seen.append)

    for num in range(2, 100):
        frames[num], frames[num - 1], frames[num - 2]

    assert seen == sorted(seen)


def test_progress_frames_behaves_like_a_list():
    from src.pipeline import _ProgressFrames

    frames = _ProgressFrames([10, 20, 30], lambda _: None)

    assert len(frames) == 3
    assert frames[-1] == 30
    assert frames[0:2] == [10, 20]


def test_progress_frames_handles_an_empty_list():
    from src.pipeline import _ProgressFrames

    seen = []
    assert list(_ProgressFrames([], seen.append)) == []
    assert seen == []


# --- Cache location --------------------------------------------------------------------------

def test_cache_dir_is_derived_from_the_clip_name():
    assert cache_dir_for("data/samples/xenia_720p.mp4").name == "xenia_720p"
    assert cache_dir_for("clip.mp4", root="/tmp/x").parent.as_posix() == "/tmp/x"
