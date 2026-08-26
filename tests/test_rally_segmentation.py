import pytest

from src.analysis.rally_segmentation import (
    MAX_PROPOSAL_CONFIDENCE,
    Rally,
    propose_outcome,
    propose_outcomes,
    segment_rallies,
)
from src.scoring import rules

FPS = 30.0


def track_with(positions):
    """Build a frame-indexed court-position list from {frame: (x, y)}."""
    length = max(positions) + 1
    track = [None] * length
    for frame, point in positions.items():
        track[frame] = point
    return track


# Points comfortably inside each half, and clearly outside.
NEAR_HALF_IN = (rules.CENTER_X_M, rules.NET_Y_M + 4.0)  # player 0's side
FAR_HALF_IN = (rules.CENTER_X_M, rules.NET_Y_M - 4.0)  # player 1's side
NEAR_HALF_OUT = (rules.CENTER_X_M, rules.DOUBLES_LENGTH_M + 1.5)  # long past player 0's baseline
FAR_HALF_OUT = (rules.CENTER_X_M, -1.5)  # long past player 1's baseline


# --- segment_rallies ------------------------------------------------------------------

def test_segment_rallies_groups_closely_spaced_bounces():
    bounces = [10, 40, 70, 100]  # 1s apart at 30fps

    rallies = segment_rallies(bounces, FPS)

    assert len(rallies) == 1
    assert rallies[0].bounce_frames == (10, 40, 70, 100)
    assert rallies[0].start_frame == 10
    assert rallies[0].end_frame == 100


def test_segment_rallies_splits_on_a_long_gap():
    # 5 seconds of quiet between the two clusters.
    bounces = [10, 40, 70, 250, 280]

    rallies = segment_rallies(bounces, FPS)

    assert len(rallies) == 2
    assert rallies[0].bounce_frames == (10, 40, 70)
    assert rallies[1].bounce_frames == (250, 280)


def test_segment_rallies_gap_threshold_is_configurable():
    bounces = [0, 90]  # exactly 3s apart

    assert len(segment_rallies(bounces, FPS, max_gap_s=2.0)) == 2
    assert len(segment_rallies(bounces, FPS, max_gap_s=4.0)) == 1


def test_segment_rallies_handles_empty_and_single_bounce():
    assert segment_rallies([], FPS) == []

    rallies = segment_rallies([42], FPS)
    assert len(rallies) == 1
    assert rallies[0].start_frame == rallies[0].end_frame == 42


def test_segment_rallies_sorts_and_deduplicates():
    rallies = segment_rallies([70, 10, 40, 40], FPS)

    assert len(rallies) == 1
    assert rallies[0].bounce_frames == (10, 40, 70)


def test_rally_reports_duration_and_bounce_count():
    rally = Rally(start_frame=30, end_frame=120, bounce_frames=(30, 75, 120))

    assert rally.n_bounces == 3
    assert rally.duration_s(FPS) == pytest.approx(3.0)


# --- propose_outcome: who won ---------------------------------------------------------

def test_ball_landing_out_on_near_side_gives_the_point_to_the_near_player():
    # Player 1 (far) hit it long past player 0's baseline -> player 0 wins.
    rally = Rally(start_frame=0, end_frame=10, bounce_frames=(10,))
    track = track_with({10: NEAR_HALF_OUT})

    outcome = propose_outcome(rally, track)

    assert outcome.winner == 0
    assert outcome.was_in is False
    assert "out" in outcome.reason


def test_ball_landing_out_on_far_side_gives_the_point_to_the_far_player():
    rally = Rally(start_frame=0, end_frame=10, bounce_frames=(10,))
    track = track_with({10: FAR_HALF_OUT})

    outcome = propose_outcome(rally, track)

    assert outcome.winner == 1
    assert outcome.was_in is False


def test_ball_landing_in_gives_the_point_to_the_hitter():
    # Landed in on player 0's side -> hit by player 1, and player 0 failed to return.
    rally = Rally(start_frame=0, end_frame=10, bounce_frames=(10,))
    track = track_with({10: NEAR_HALF_IN})

    outcome = propose_outcome(rally, track)

    assert outcome.winner == 1
    assert outcome.was_in is True
    assert "failed to return" in outcome.reason


def test_only_the_final_bounce_decides_the_outcome():
    # Earlier bounces are in play; the last one ends the point.
    rally = Rally(start_frame=0, end_frame=90, bounce_frames=(10, 50, 90))
    track = track_with({10: NEAR_HALF_IN, 50: FAR_HALF_IN, 90: NEAR_HALF_OUT})

    outcome = propose_outcome(rally, track)

    assert outcome.bounce_frame == 90
    assert outcome.winner == 0


def test_falls_back_to_the_last_bounce_that_has_a_position():
    rally = Rally(start_frame=0, end_frame=90, bounce_frames=(10, 50, 90))
    track = track_with({10: NEAR_HALF_IN, 50: NEAR_HALF_OUT})  # frame 90 never tracked

    outcome = propose_outcome(rally, track)

    assert outcome.bounce_frame == 50
    assert outcome.winner == 0


def test_no_usable_bounce_yields_no_proposal():
    rally = Rally(start_frame=0, end_frame=90, bounce_frames=(10, 50, 90))
    track = [None] * 100

    outcome = propose_outcome(rally, track)

    assert outcome.winner is None
    assert outcome.confidence == 0.0
    assert "no ball position" in outcome.reason


# --- Borrowing a position from a nearby frame ------------------------------------------
# The bounce detector interpolates internally, so it reports bounces on frames our
# cached track has as None. Without this search, whole real rallies were unscoreable.

def test_position_is_borrowed_from_a_neighbouring_frame():
    rally = Rally(start_frame=0, end_frame=50, bounce_frames=(50,))
    track = track_with({48: NEAR_HALF_OUT})  # nothing tracked at frame 50 itself

    outcome = propose_outcome(rally, track)

    assert outcome.winner == 0
    assert outcome.bounce_frame == 50
    assert "2 frame(s) away" in outcome.reason


def test_borrowed_position_lowers_confidence():
    rally = Rally(start_frame=0, end_frame=50, bounce_frames=(50,))
    exact = propose_outcome(rally, track_with({50: NEAR_HALF_OUT}))
    borrowed = propose_outcome(rally, track_with({47: NEAR_HALF_OUT}))

    assert borrowed.winner == exact.winner
    assert borrowed.confidence < exact.confidence


def test_nearest_frame_wins_when_positions_exist_on_both_sides():
    rally = Rally(start_frame=0, end_frame=50, bounce_frames=(50,))
    # Frame 49 is nearer than 47, and they'd give opposite rulings.
    track = track_with({47: FAR_HALF_OUT, 49: NEAR_HALF_OUT})

    outcome = propose_outcome(rally, track)

    assert outcome.winner == 0, "should use frame 49, not 47"


def test_position_search_can_be_disabled():
    rally = Rally(start_frame=0, end_frame=50, bounce_frames=(50,))
    track = track_with({48: NEAR_HALF_OUT})

    assert propose_outcome(rally, track, position_search_frames=0).winner is None


def test_position_search_does_not_reach_beyond_its_window():
    rally = Rally(start_frame=0, end_frame=50, bounce_frames=(50,))
    track = track_with({40: NEAR_HALF_OUT})  # 10 frames away, outside the default radius

    assert propose_outcome(rally, track).winner is None


def test_bounce_frame_beyond_the_track_is_ignored_safely():
    rally = Rally(start_frame=0, end_frame=500, bounce_frames=(500,))
    track = track_with({10: NEAR_HALF_IN})

    outcome = propose_outcome(rally, track)

    assert outcome.winner is None


# --- propose_outcome: confidence discipline --------------------------------------------

def test_confidence_is_capped_below_certainty_even_for_an_obvious_call():
    """A geometrically certain in/out call still can't prove why the rally ended."""
    rally = Rally(start_frame=0, end_frame=10, bounce_frames=(10,))
    track = track_with({10: (rules.CENTER_X_M, -5.0)})  # miles out

    outcome = propose_outcome(rally, track)

    assert outcome.confidence <= MAX_PROPOSAL_CONFIDENCE
    assert outcome.confidence > 0


def test_near_line_bounce_is_less_confident_than_a_clear_one():
    rally = Rally(start_frame=0, end_frame=10, bounce_frames=(10,))
    on_the_line = track_with({10: (rules.LEFT_SINGLES_X_M, rules.NET_Y_M + 4.0)})
    well_inside = track_with({10: NEAR_HALF_IN})

    borderline = propose_outcome(rally, on_the_line)
    clear = propose_outcome(rally, well_inside)

    assert borderline.confidence < clear.confidence


def test_singles_and_doubles_can_disagree_on_the_same_bounce():
    # A bounce in the doubles alley: out in singles, in for doubles -- and the two
    # rulings hand the point to opposite players.
    alley = (rules.LEFT_SINGLES_X_M / 2, rules.NET_Y_M + 4.0)
    rally = Rally(start_frame=0, end_frame=10, bounce_frames=(10,))
    track = track_with({10: alley})

    singles = propose_outcome(rally, track, court_type="singles")
    doubles = propose_outcome(rally, track, court_type="doubles")

    assert singles.winner == 0
    assert doubles.winner == 1


# --- propose_outcomes -------------------------------------------------------------------

def test_propose_outcomes_pairs_each_rally_with_its_proposal():
    rallies = segment_rallies([10, 40, 250], FPS)
    track = track_with({10: NEAR_HALF_IN, 40: NEAR_HALF_OUT, 250: FAR_HALF_OUT})

    results = propose_outcomes(rallies, track)

    assert len(results) == 2
    assert [outcome.winner for _, outcome in results] == [0, 1]
    assert all(rally is pair[0] for rally, pair in zip(rallies, results))
