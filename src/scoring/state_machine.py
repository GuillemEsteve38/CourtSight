"""Tennis scoring state machine: points, games, sets, tiebreaks, serve rotation.

Pure logic -- no ML, no GPU, no video. Every transition is a function of the current
state plus "who won the point", which makes the whole thing exhaustively testable and
means a wrong automated call can never corrupt the score beyond what a human can undo.

Players are indexed 0 and 1. By the convention used in src/analysis/rally_segmentation.py,
player 0 is the near (bottom-of-frame) player and player 1 is the far one.

MatchState is immutable: award_point() returns a new state rather than mutating. That is
what makes MatchScorer's undo trivial (pop a stack) -- and undo is a hard requirement
here, because rally outcomes are proposed by unreliable automation and corrected by a
human (see development-plan.md's hybrid confirmation).
"""

from dataclasses import dataclass, field, replace

POINT_NAMES = ("0", "15", "30", "40")
ADVANTAGE = "AD"

# A standard game needs 4 points; a tiebreak's target lives in MatchRules.
POINTS_TO_WIN_GAME = 4
WIN_BY_MARGIN = 2


@dataclass(frozen=True)
class MatchRules:
    """Format knobs. Defaults are a standard best-of-3 with tiebreaks in every set."""

    best_of: int = 3
    games_to_win_set: int = 6
    tiebreak_at_games: int = 6  # games level at this count -> tiebreak
    tiebreak_points: int = 7
    final_set_tiebreak: bool = True  # False -> advantage (unlimited) final set
    ad_scoring: bool = True  # False -> no-ad, sudden-death point at deuce

    @property
    def sets_to_win_match(self):
        return self.best_of // 2 + 1


@dataclass(frozen=True)
class MatchState:
    """A complete, immutable snapshot of the score."""

    points: tuple = (0, 0)  # raw point count in the current game/tiebreak
    games: tuple = (0, 0)  # games in the current set
    sets: tuple = (0, 0)  # sets won
    game_server: int = 0  # who serves this game; in a tiebreak, who served its 1st point
    completed_sets: tuple = ()  # ((games0, games1), ...) for finished sets
    winner: int = None  # set once the match is decided
    rules: MatchRules = field(default_factory=MatchRules)

    # --- Derived views ---------------------------------------------------------------

    @property
    def is_final_set(self):
        """True in the deciding set (1-1 in a best-of-3, 2-2 in a best-of-5)."""
        return sum(self.sets) == self.rules.best_of - 1

    @property
    def in_tiebreak(self):
        if self.winner is not None:
            return False
        at = self.rules.tiebreak_at_games
        if self.games != (at, at):
            return False
        return not (self.is_final_set and not self.rules.final_set_tiebreak)

    @property
    def server(self):
        """Who serves the next point.

        Normally the game's server. In a tiebreak serve changes after the first point
        and every two points thereafter, so it is derived from points played rather
        than stored -- storing it would let it drift out of sync with the score.
        """
        if not self.in_tiebreak:
            return self.game_server
        alternations = (sum(self.points) + 1) // 2
        return self.game_server if alternations % 2 == 0 else 1 - self.game_server

    def point_score_display(self):
        """Points as they'd be called aloud: ('30', '40'), ('AD', '40'), ('5', '3')."""
        if self.in_tiebreak:
            return tuple(str(p) for p in self.points)

        first, second = self.points
        if self.rules.ad_scoring and first >= 3 and second >= 3:
            if first == second:
                return (POINT_NAMES[3], POINT_NAMES[3])  # deuce
            return (ADVANTAGE, POINT_NAMES[3]) if first > second else (POINT_NAMES[3], ADVANTAGE)
        return (POINT_NAMES[min(first, 3)], POINT_NAMES[min(second, 3)])

    @property
    def is_deuce(self):
        return (
            self.rules.ad_scoring
            and not self.in_tiebreak
            and self.points[0] >= 3
            and self.points[0] == self.points[1]
        )

    def __str__(self):
        sets_part = " ".join(f"{a}-{b}" for a, b in self.completed_sets)
        games_part = f"{self.games[0]}-{self.games[1]}"
        points_part = "-".join(self.point_score_display())
        pieces = [p for p in (sets_part, games_part, points_part) if p]
        summary = " | ".join(pieces)
        if self.winner is not None:
            return f"{sets_part} | player {self.winner} wins"
        return f"{summary} (server: player {self.server})"

    # --- Transition ------------------------------------------------------------------

    def award_point(self, player):
        """Return the state after `player` wins a point. Never mutates self."""
        if self.winner is not None:
            raise ValueError("match is already complete; undo before scoring again")
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player!r}")

        points = list(self.points)
        points[player] += 1

        if self._wins_game(points, player):
            return self._award_game(player)
        return replace(self, points=tuple(points))

    def _wins_game(self, points, player):
        opponent = 1 - player
        if self.in_tiebreak:
            return (
                points[player] >= self.rules.tiebreak_points
                and points[player] - points[opponent] >= WIN_BY_MARGIN
            )
        if not self.rules.ad_scoring:
            return points[player] >= POINTS_TO_WIN_GAME
        return (
            points[player] >= POINTS_TO_WIN_GAME
            and points[player] - points[opponent] >= WIN_BY_MARGIN
        )

    def _award_game(self, player):
        games = list(self.games)
        games[player] += 1

        # Serve always passes to the other player for the next game. After a tiebreak
        # this also yields the correct result: whoever served the tiebreak's first
        # point receives first in the next set.
        next_state = replace(
            self,
            points=(0, 0),
            games=tuple(games),
            game_server=1 - self.game_server,
        )

        if self.in_tiebreak or _set_is_won(games, player, self.rules):
            return next_state._award_set(player)
        return next_state

    def _award_set(self, player):
        sets = list(self.sets)
        sets[player] += 1

        next_state = replace(
            self,
            sets=tuple(sets),
            completed_sets=self.completed_sets + (self.games,),
            games=(0, 0),
            points=(0, 0),
        )

        if sets[player] >= self.rules.sets_to_win_match:
            return replace(next_state, winner=player)
        return next_state


def _set_is_won(games, player, rules):
    opponent = 1 - player
    return (
        games[player] >= rules.games_to_win_set
        and games[player] - games[opponent] >= WIN_BY_MARGIN
    )


@dataclass(frozen=True)
class PointRecord:
    """One scored point, kept so the match can be audited and replayed."""

    winner: int
    source: str  # "manual", "auto", or "corrected"
    proposed_winner: int = None
    confidence: float = None

    @property
    def was_corrected(self):
        return self.source == "corrected"


class MatchScorer:
    """Stateful wrapper around MatchState adding history, undo, and the confirm hook.

    The hybrid flow from development-plan.md: automation proposes a rally outcome,
    a human confirms or overrides it, and the *human's* answer drives the state
    update. Corrections are recorded rather than silently applied, so the proportion
    of overrides is measurable -- that number is the honest measure of how good the
    automatic outcome detection actually is.
    """

    def __init__(self, rules=None, first_server=0):
        if first_server not in (0, 1):
            raise ValueError(f"first_server must be 0 or 1, got {first_server!r}")
        initial = MatchState(rules=rules or MatchRules(), game_server=first_server)
        self._history = [initial]
        self._points = []

    @property
    def state(self):
        return self._history[-1]

    @property
    def points_played(self):
        return tuple(self._points)

    def award_point(self, winner, source="manual", proposed_winner=None, confidence=None):
        """Score a point outright, bypassing the propose/confirm flow."""
        self._history.append(self.state.award_point(winner))
        self._points.append(
            PointRecord(
                winner=winner,
                source=source,
                proposed_winner=proposed_winner,
                confidence=confidence,
            )
        )
        return self.state

    def apply_outcome(self, outcome, confirmed_winner=None):
        """Apply a proposed rally outcome, optionally overridden by a human.

        :param outcome: anything exposing .winner (and optionally .confidence) --
            duck-typed so src/scoring/ stays independent of src/analysis/
        :param confirmed_winner: the human's answer; overrides outcome.winner
        """
        proposed = outcome.winner
        winner = proposed if confirmed_winner is None else confirmed_winner
        if winner is None:
            raise ValueError(
                "no winner: the proposal was inconclusive and needs confirmed_winner"
            )

        if confirmed_winner is None:
            source = "auto"
        elif proposed is None or confirmed_winner == proposed:
            source = "manual" if proposed is None else "auto"
        else:
            source = "corrected"

        return self.award_point(
            winner,
            source=source,
            proposed_winner=proposed,
            confidence=getattr(outcome, "confidence", None),
        )

    def undo(self):
        """Step back one point. Essential: automated calls will be wrong sometimes."""
        if len(self._history) == 1:
            raise ValueError("nothing to undo")
        self._history.pop()
        self._points.pop()
        return self.state

    @property
    def correction_rate(self):
        """Fraction of scored points a human had to override. None if nothing scored."""
        if not self._points:
            return None
        corrected = sum(1 for p in self._points if p.was_corrected)
        return corrected / len(self._points)
