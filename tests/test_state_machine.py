import pytest

from src.scoring.state_machine import (
    MatchRules,
    MatchScorer,
    MatchState,
)


def play(state, winners):
    """Award a sequence of points, e.g. play(state, [0, 0, 1, 0, 0])."""
    for winner in winners:
        state = state.award_point(winner)
    return state


def win_game(state, player):
    """Win a full game from the current point score (assumes not already in deuce)."""
    while state.games == state.games:  # loop until the game count changes
        before = state.games
        state = state.award_point(player)
        if state.games != before or state.winner is not None:
            return state
    return state


def win_games(state, player, count):
    for _ in range(count):
        state = win_game(state, player)
    return state


# --- Point progression ---------------------------------------------------------------

def test_points_progress_through_15_30_40():
    state = MatchState()

    assert state.point_score_display() == ("0", "0")
    state = state.award_point(0)
    assert state.point_score_display() == ("15", "0")
    state = state.award_point(0)
    assert state.point_score_display() == ("30", "0")
    state = state.award_point(0)
    assert state.point_score_display() == ("40", "0")


def test_winning_fourth_point_takes_the_game():
    state = play(MatchState(), [0, 0, 0, 0])

    assert state.games == (1, 0)
    assert state.points == (0, 0)  # reset for the next game


def test_game_requires_a_two_point_margin():
    state = play(MatchState(), [0, 0, 0, 1])  # 40-15
    state = state.award_point(1)  # 40-30
    assert state.games == (0, 0)
    state = state.award_point(0)  # game
    assert state.games == (1, 0)


# --- Deuce and advantage ---------------------------------------------------------------

def test_deuce_is_reached_at_three_points_each():
    state = play(MatchState(), [0, 1, 0, 1, 0, 1])

    assert state.is_deuce
    assert state.point_score_display() == ("40", "40")
    assert state.games == (0, 0)


def test_advantage_then_back_to_deuce():
    state = play(MatchState(), [0, 1, 0, 1, 0, 1])  # deuce

    state = state.award_point(0)
    assert state.point_score_display() == ("AD", "40")

    state = state.award_point(1)
    assert state.is_deuce, "losing the advantage point returns to deuce"
    assert state.point_score_display() == ("40", "40")


def test_two_points_from_deuce_wins_the_game():
    state = play(MatchState(), [0, 1, 0, 1, 0, 1])  # deuce
    state = play(state, [1, 1])

    assert state.games == (0, 1)


def test_long_deuce_battle_resolves_correctly():
    state = play(MatchState(), [0, 1, 0, 1, 0, 1])  # deuce
    # Five swings of advantage back to deuce, then player 1 closes it out.
    state = play(state, [0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    assert state.is_deuce
    assert state.games == (0, 0)

    state = play(state, [1, 1])
    assert state.games == (0, 1)


def test_no_ad_scoring_decides_at_deuce():
    state = MatchState(rules=MatchRules(ad_scoring=False))
    state = play(state, [0, 1, 0, 1, 0, 1])  # 40-40, sudden death

    assert state.games == (0, 0)
    state = state.award_point(1)
    assert state.games == (0, 1), "no-ad: next point takes the game outright"


# --- Sets --------------------------------------------------------------------------------

def test_set_won_at_six_love():
    state = win_games(MatchState(), 0, 6)

    assert state.sets == (1, 0)
    assert state.completed_sets == ((6, 0),)
    assert state.games == (0, 0)


def test_set_requires_a_two_game_margin():
    state = MatchState()
    for _ in range(5):
        state = win_game(state, 0)
        state = win_game(state, 1)
    # 5-5
    assert state.games == (5, 5)

    state = win_game(state, 0)  # 6-5 -- not yet a set
    assert state.sets == (0, 0)
    assert state.games == (6, 5)

    state = win_game(state, 0)  # 7-5
    assert state.sets == (1, 0)
    assert state.completed_sets == ((7, 5),)


def test_six_four_takes_the_set():
    state = MatchState()
    for _ in range(4):
        state = win_game(state, 0)
        state = win_game(state, 1)
    state = win_games(state, 0, 2)  # 6-4

    assert state.completed_sets == ((6, 4),)


# --- Tiebreak -------------------------------------------------------------------------------

def reach_six_all(state=None):
    state = state or MatchState()
    for _ in range(6):
        state = win_game(state, 0)
        state = win_game(state, 1)
    return state


def test_tiebreak_starts_at_six_all():
    state = reach_six_all()

    assert state.games == (6, 6)
    assert state.in_tiebreak
    assert state.point_score_display() == ("0", "0")


def test_tiebreak_points_are_displayed_numerically():
    state = play(reach_six_all(), [0, 0, 1])

    assert state.point_score_display() == ("2", "1")


def test_tiebreak_won_at_seven_points_takes_the_set():
    state = play(reach_six_all(), [0] * 7)

    assert state.sets == (1, 0)
    assert state.completed_sets == ((7, 6),), "tiebreak set is recorded as 7-6"


def test_tiebreak_requires_a_two_point_margin():
    state = reach_six_all()
    state = play(state, [0, 1] * 6)  # 6-6 in the tiebreak
    assert state.sets == (0, 0)

    state = state.award_point(0)  # 7-6, still not enough
    assert state.sets == (0, 0)

    state = state.award_point(0)  # 8-6
    assert state.sets == (1, 0)


def test_tiebreak_serve_rotation():
    # One point on the opening serve, then two each -- the standard rotation.
    state = reach_six_all()
    first_server = state.game_server
    other = 1 - first_server

    expected = [first_server, other, other, first_server, first_server, other, other]
    actual = []
    for _ in expected:
        actual.append(state.server)
        state = state.award_point(0)

    assert actual == expected


def test_server_after_tiebreak_is_the_tiebreak_receiver():
    state = reach_six_all()
    tiebreak_first_server = state.game_server

    state = play(state, [0] * 7)

    assert state.game_server == 1 - tiebreak_first_server


def test_advantage_final_set_has_no_tiebreak():
    rules = MatchRules(best_of=3, final_set_tiebreak=False)
    state = MatchState(rules=rules, sets=(1, 1))  # deciding set
    state = reach_six_all(state)

    assert state.is_final_set
    assert not state.in_tiebreak, "advantage set keeps playing games past 6-6"

    state = win_game(state, 0)  # 7-6
    assert state.sets == (1, 1), "still no set winner without a two-game margin"
    state = win_game(state, 0)  # 8-6
    assert state.sets == (2, 1)


def test_non_final_set_still_uses_tiebreak_when_final_set_tiebreak_disabled():
    rules = MatchRules(best_of=3, final_set_tiebreak=False)
    state = reach_six_all(MatchState(rules=rules))  # first set, 0-0 in sets

    assert not state.is_final_set
    assert state.in_tiebreak


# --- Serve rotation ---------------------------------------------------------------------------

def test_serve_alternates_every_game():
    state = MatchState(game_server=0)

    assert state.server == 0
    state = win_game(state, 0)
    assert state.server == 1
    state = win_game(state, 1)
    assert state.server == 0


def test_serve_keeps_alternating_across_a_set_boundary():
    state = win_games(MatchState(game_server=0), 0, 6)  # set won, 6 games played

    assert state.sets == (1, 0)
    assert state.server == 0, "6 games played, so serve returns to the original server"


# --- Match completion ----------------------------------------------------------------------------

def test_best_of_three_needs_two_sets():
    state = MatchState()
    state = win_games(state, 0, 6)
    assert state.winner is None

    state = win_games(state, 0, 6)
    assert state.winner == 0
    assert state.sets == (2, 0)


def test_best_of_five_needs_three_sets():
    state = MatchState(rules=MatchRules(best_of=5))
    for _ in range(2):
        state = win_games(state, 0, 6)
    assert state.winner is None

    state = win_games(state, 0, 6)
    assert state.winner == 0


def test_scoring_after_match_end_is_rejected():
    state = win_games(win_games(MatchState(), 0, 6), 0, 6)

    with pytest.raises(ValueError):
        state.award_point(0)


def test_invalid_player_rejected():
    with pytest.raises(ValueError):
        MatchState().award_point(2)


# --- Full match simulation -------------------------------------------------------------------------

def test_full_three_set_match_with_deuce_and_tiebreak():
    """End-to-end: player 1 loses the first set, wins a tiebreak, takes the decider."""
    state = MatchState()

    # Set 1: player 0 wins 6-3.
    for _ in range(3):
        state = win_game(state, 0)
        state = win_game(state, 1)
    state = win_games(state, 0, 3)
    assert state.completed_sets == ((6, 3),)
    assert state.sets == (1, 0)

    # Set 2: reaches 6-6, player 1 takes the tiebreak 7-5.
    state = reach_six_all(state)
    assert state.in_tiebreak
    state = play(state, [1, 0] * 5)  # 5-5
    state = play(state, [1, 1])  # 7-5
    assert state.sets == (1, 1)
    # Games are (player0, player1), so player 1 winning the tiebreak reads 6-7.
    assert state.completed_sets == ((6, 3), (6, 7))

    # Set 3 (decider): player 1 wins 6-4, taking one game through deuce.
    assert state.is_final_set
    state = play(state, [1, 0, 1, 0, 1, 0])  # deuce in the opening game
    assert state.is_deuce
    state = play(state, [1, 1])  # player 1 holds the opening game
    assert state.games == (0, 1)

    for _ in range(4):
        state = win_game(state, 0)
        state = win_game(state, 1)
    # games now 4-5
    state = win_game(state, 1)  # 4-6

    assert state.winner == 1
    assert state.sets == (1, 2)
    assert state.completed_sets == ((6, 3), (6, 7), (4, 6))


# --- MatchScorer: history, undo, hybrid confirmation -------------------------------------------------

class FakeOutcome:
    def __init__(self, winner, confidence=0.5):
        self.winner = winner
        self.confidence = confidence


def test_scorer_tracks_state_and_history():
    scorer = MatchScorer()

    scorer.award_point(0)
    scorer.award_point(1)

    assert scorer.state.points == (1, 1)
    assert len(scorer.points_played) == 2


def test_scorer_undo_restores_previous_state():
    scorer = MatchScorer()
    scorer.award_point(0)
    scorer.award_point(0)
    scorer.award_point(0)
    scorer.award_point(0)
    assert scorer.state.games == (1, 0)

    scorer.undo()

    assert scorer.state.games == (0, 0)
    assert scorer.state.points == (3, 0)
    assert len(scorer.points_played) == 3


def test_scorer_undo_past_start_is_rejected():
    with pytest.raises(ValueError):
        MatchScorer().undo()


def test_undo_recovers_from_a_wrong_automatic_call():
    """The whole reason MatchState is immutable: bad auto-calls must be reversible."""
    scorer = MatchScorer()
    scorer.apply_outcome(FakeOutcome(winner=0))
    scorer.apply_outcome(FakeOutcome(winner=0))
    assert scorer.state.points == (2, 0)

    scorer.undo()
    scorer.apply_outcome(FakeOutcome(winner=0), confirmed_winner=1)

    assert scorer.state.points == (1, 1)


def test_apply_outcome_uses_proposal_when_not_overridden():
    scorer = MatchScorer()

    scorer.apply_outcome(FakeOutcome(winner=1))

    assert scorer.state.points == (0, 1)
    record = scorer.points_played[0]
    assert record.source == "auto"
    assert record.proposed_winner == 1
    assert not record.was_corrected


def test_apply_outcome_records_a_human_override():
    scorer = MatchScorer()

    scorer.apply_outcome(FakeOutcome(winner=1), confirmed_winner=0)

    assert scorer.state.points == (1, 0), "the human's answer drives the state"
    record = scorer.points_played[0]
    assert record.was_corrected
    assert record.proposed_winner == 1
    assert record.winner == 0


def test_confirming_a_correct_proposal_is_not_a_correction():
    scorer = MatchScorer()

    scorer.apply_outcome(FakeOutcome(winner=1), confirmed_winner=1)

    assert not scorer.points_played[0].was_corrected


def test_inconclusive_proposal_requires_a_human_answer():
    scorer = MatchScorer()

    with pytest.raises(ValueError):
        scorer.apply_outcome(FakeOutcome(winner=None))

    scorer.apply_outcome(FakeOutcome(winner=None), confirmed_winner=0)
    assert scorer.state.points == (1, 0)
    assert scorer.points_played[0].source == "manual"


def test_correction_rate_measures_automation_quality():
    scorer = MatchScorer()
    assert scorer.correction_rate is None

    scorer.apply_outcome(FakeOutcome(winner=0))  # accepted
    scorer.apply_outcome(FakeOutcome(winner=0))  # accepted
    scorer.apply_outcome(FakeOutcome(winner=0), confirmed_winner=1)  # corrected

    assert scorer.correction_rate == pytest.approx(1 / 3)


def test_scorer_rejects_bad_first_server():
    with pytest.raises(ValueError):
        MatchScorer(first_server=2)
