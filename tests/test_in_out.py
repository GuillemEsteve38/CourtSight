import pytest

from src.analysis.in_out import (
    DEFAULT_TOLERANCE_M,
    MIN_CONFIDENCE,
    classify_bounce,
)
from src.scoring import rules

# --- Rally shots: singles vs. doubles court width ---------------------------------

def test_rally_singles_clearly_in():
    point = (rules.CENTER_X_M, 5.0)  # dead center, well inside either court
    call = classify_bounce(point, shot_type="rally", court_type="singles")
    assert call.is_in
    assert call.confidence == pytest.approx(1.0)


def test_rally_doubles_alley_is_in_for_doubles_but_out_for_singles():
    # Halfway between the singles and doubles sideline: inside the doubles alley.
    alley_x = (rules.LEFT_SINGLES_X_M + 0.0) / 2
    point = (alley_x, 5.0)

    singles_call = classify_bounce(point, shot_type="rally", court_type="singles")
    doubles_call = classify_bounce(point, shot_type="rally", court_type="doubles")

    assert not singles_call.is_in
    assert doubles_call.is_in


def test_rally_clearly_out_far_beyond_doubles_sideline():
    point = (-2.0, 5.0)  # 2m past the doubles sideline
    call = classify_bounce(point, shot_type="rally", court_type="doubles")
    assert not call.is_in
    assert call.confidence == pytest.approx(1.0)
    assert call.distance_m < 0


# --- Near-line calls (tolerance band) ----------------------------------------------

def test_point_exactly_on_sideline_is_in():
    point = (rules.LEFT_SINGLES_X_M, 5.0)
    call = classify_bounce(point, shot_type="rally", court_type="singles")
    assert call.is_in
    assert call.confidence == pytest.approx(MIN_CONFIDENCE)


def test_point_just_outside_within_tolerance_is_in_with_low_confidence():
    just_outside = DEFAULT_TOLERANCE_M * 0.5
    point = (rules.LEFT_SINGLES_X_M - just_outside, 5.0)
    call = classify_bounce(point, shot_type="rally", court_type="singles")
    assert call.is_in
    assert MIN_CONFIDENCE < call.confidence < 1.0


def test_point_just_outside_beyond_tolerance_is_out():
    beyond_tolerance = DEFAULT_TOLERANCE_M + 0.01
    point = (rules.LEFT_SINGLES_X_M - beyond_tolerance, 5.0)
    call = classify_bounce(point, shot_type="rally", court_type="singles")
    assert not call.is_in


def test_baseline_near_line_calls():
    on_baseline = classify_bounce((rules.CENTER_X_M, 0.0), shot_type="rally", court_type="singles")
    assert on_baseline.is_in
    assert on_baseline.confidence == pytest.approx(MIN_CONFIDENCE)

    slightly_long = classify_bounce(
        (rules.CENTER_X_M, -0.20), shot_type="rally", court_type="singles"
    )
    assert not slightly_long.is_in
    assert slightly_long.confidence == pytest.approx(1.0)


# --- Serve shots: service boxes ----------------------------------------------------

@pytest.mark.parametrize("box_name", sorted(rules.SERVICE_BOXES))
def test_serve_lands_in_correct_box(box_name):
    box = rules.SERVICE_BOXES[box_name]
    center = ((box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2)
    call = classify_bounce(center, shot_type="serve", service_box=box_name)
    assert call.is_in
    assert call.confidence == pytest.approx(1.0)


def test_serve_long_past_service_line_is_out():
    # Well past the top service box, deep into the backcourt -- would be "in" for a
    # rally shot, but not for a serve.
    point = (rules.CENTER_X_M, 2.0)
    call = classify_bounce(point, shot_type="serve", service_box="top_left")
    assert not call.is_in

    rally_call = classify_bounce(point, shot_type="rally", court_type="singles")
    assert rally_call.is_in


def test_serve_in_doubles_alley_is_out_even_though_alley_is_in_for_doubles_rally():
    alley_x = rules.LEFT_SINGLES_X_M / 2
    point = (alley_x, (rules.TOP_SERVICE_Y_M + rules.NET_Y_M) / 2)
    call = classify_bounce(point, shot_type="serve", service_box="top_left")
    assert not call.is_in


def test_serve_requires_valid_service_box():
    with pytest.raises(ValueError):
        classify_bounce((5.0, 5.0), shot_type="serve", service_box=None)
    with pytest.raises(ValueError):
        classify_bounce((5.0, 5.0), shot_type="serve", service_box="middle")


# --- Input validation ---------------------------------------------------------------

def test_invalid_shot_type_raises():
    with pytest.raises(ValueError):
        classify_bounce((5.0, 5.0), shot_type="smash")


def test_invalid_court_type_raises():
    with pytest.raises(ValueError):
        classify_bounce((5.0, 5.0), shot_type="rally", court_type="mixed")


# --- Confidence combination with tracking quality -----------------------------------

def test_tracking_confidence_scales_final_confidence():
    point = (rules.CENTER_X_M, 5.0)  # clearly in -> geometric confidence 1.0
    full = classify_bounce(point, shot_type="rally", court_type="singles", tracking_confidence=1.0)
    half = classify_bounce(point, shot_type="rally", court_type="singles", tracking_confidence=0.5)

    assert full.confidence == pytest.approx(1.0)
    assert half.confidence == pytest.approx(0.5)
    assert full.is_in and half.is_in  # tracking confidence doesn't flip the call, only the score


def test_confidence_increases_with_distance_from_line():
    near = classify_bounce(
        (rules.LEFT_SINGLES_X_M + 0.01, 5.0), shot_type="rally", court_type="singles"
    )
    far = classify_bounce(
        (rules.LEFT_SINGLES_X_M + 1.0, 5.0), shot_type="rally", court_type="singles"
    )
    assert near.confidence < far.confidence
    assert far.confidence == pytest.approx(1.0)
