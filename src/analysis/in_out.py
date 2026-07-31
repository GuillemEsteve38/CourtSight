"""In/out line-calling from a bounce point in court meters.

Given where the ball bounced (in the court-meters coordinate system produced by
src/perception/homography.py) plus shot context (serve vs. rally, singles vs.
doubles, and which service box for serves), decide in/out against the court
reference model in src/scoring/rules.py, with a tolerance band for line thickness /
homography measurement slack and a confidence score.
"""

from dataclasses import dataclass

from src.scoring import rules

DEFAULT_TOLERANCE_M = 0.05  # ~court line width + homography measurement slack
CONFIDENCE_FULL_CERTAINTY_MARGIN_M = 0.10  # beyond this margin from the line, confidence maxes out
MIN_CONFIDENCE = 0.5  # right on the line: a coin flip, never below this


@dataclass(frozen=True)
class InOutCall:
    is_in: bool
    distance_m: float  # signed: positive = inside the valid region, negative = outside
    confidence: float  # MIN_CONFIDENCE (right on the line) .. 1.0 (clearly in or out)


def classify_bounce(
    point_m,
    shot_type,
    court_type="singles",
    service_box=None,
    tolerance_m=DEFAULT_TOLERANCE_M,
    tracking_confidence=1.0,
):
    """Classify a bounce point as in or out.

    :param point_m: (x, y) bounce point in court meters
    :param shot_type: "serve" or "rally"
    :param court_type: "singles" or "doubles" -- only affects "rally" calls; serves
        always target a service box (always singles-width, regardless of match type)
    :param service_box: required when shot_type == "serve": one of
        rules.SERVICE_BOXES' keys ("top_left", "top_right", "bottom_left", "bottom_right")
    :param tolerance_m: distance outside the strict boundary that still counts as "in"
        (line thickness + measurement slack)
    :param tracking_confidence: caller-supplied [0, 1] discount for unreliable ball
        tracking near the bounce frame (e.g. gaps in the detected trajectory);
        multiplies the geometric confidence
    :return: InOutCall
    """
    region = _select_region(shot_type, court_type, service_box)
    distance = region.signed_distance(point_m)
    is_in = distance >= -tolerance_m

    certainty = min(abs(distance) / CONFIDENCE_FULL_CERTAINTY_MARGIN_M, 1.0)
    confidence = (MIN_CONFIDENCE + (1 - MIN_CONFIDENCE) * certainty) * tracking_confidence

    return InOutCall(is_in=is_in, distance_m=distance, confidence=confidence)


def _select_region(shot_type, court_type, service_box):
    if shot_type == "serve":
        if service_box not in rules.SERVICE_BOXES:
            raise ValueError(
                f"service_box must be one of {sorted(rules.SERVICE_BOXES)} for a serve, "
                f"got {service_box!r}"
            )
        return rules.SERVICE_BOXES[service_box]
    if shot_type == "rally":
        if court_type == "singles":
            return rules.SINGLES_COURT
        if court_type == "doubles":
            return rules.DOUBLES_COURT
        raise ValueError(f"court_type must be 'singles' or 'doubles', got {court_type!r}")
    raise ValueError(f"shot_type must be 'serve' or 'rally', got {shot_type!r}")
