"""Split cached tracking data into rallies and propose who won each one.

The proposals here are deliberately low-confidence. development-plan.md §8 calls
auto-detecting *why* a rally ended "research-grade", and that assessment holds: from
ball position alone we cannot see a racket miss, a net cord, or a double bounce. What
we can do is make a defensible guess from the last bounce and hand it to a human to
confirm -- which is exactly the hybrid flow MatchScorer implements.

Player convention (shared with src/scoring/state_machine.py): in the court-meters
frame from src/perception/homography.py, y runs 0 (top baseline) -> 23.77 (bottom
baseline), and the bottom of the video frame is the near player. So **player 0 is the
near/bottom player, player 1 is the far/top player**.
"""

from dataclasses import dataclass

from src.analysis.in_out import classify_bounce
from src.scoring import rules

# A rally's bounces come every second or so; a longer quiet stretch means the point
# ended and players are resetting. Tuned to be conservative -- merging two rallies is
# worse than splitting one, since a human reviewing proposals will notice a missing
# point faster than a silently merged one.
DEFAULT_MAX_BOUNCE_GAP_S = 2.5

# Ceiling on any automatic proposal. Even a geometrically certain in/out call leaves
# the "why did the rally end" inference unverified, so nothing here is ever presented
# as near-certain. Keeps the UI honest about needing a human.
MAX_PROPOSAL_CONFIDENCE = 0.6

# Bounce frames don't always coincide with tracked ball positions: the CatBoost bounce
# detector interpolates gaps internally, so it legitimately reports a bounce on a frame
# our cached track has as None. On real footage that left whole rallies unscoreable.
# Searching a couple of frames either side recovers them.
DEFAULT_POSITION_SEARCH_FRAMES = 3

# The ball moves fast near a bounce, so a position borrowed from a neighbouring frame is
# metres away from where it actually landed. Discount confidence accordingly -- at the
# edge of the search window a proposal is worth about half what an exact hit is.
MAX_OFFSET_CONFIDENCE_PENALTY = 0.5


@dataclass(frozen=True)
class Rally:
    """A candidate point: the span of play between two lulls in bounce activity."""

    start_frame: int
    end_frame: int
    bounce_frames: tuple

    @property
    def n_bounces(self):
        return len(self.bounce_frames)

    def duration_s(self, fps):
        return (self.end_frame - self.start_frame) / fps


@dataclass(frozen=True)
class RallyOutcome:
    """A proposed point winner, with the reasoning exposed for a human to judge."""

    winner: int = None  # None means "cannot tell" -- caller must decide
    reason: str = ""
    confidence: float = 0.0
    bounce_frame: int = None
    bounce_point_m: tuple = None
    was_in: bool = None


def segment_rallies(bounce_frames, fps, max_gap_s=DEFAULT_MAX_BOUNCE_GAP_S):
    """Group bounce frames into rallies, splitting on gaps longer than max_gap_s.

    Bounces are the only rally-structure signal available from cached perception
    output -- ball-presence gaps track occlusion and detection dropouts as much as
    they track play, so they'd add noise rather than information here.

    :return: list of Rally, in frame order
    """
    ordered = sorted(set(bounce_frames))
    if not ordered:
        return []

    max_gap_frames = max_gap_s * fps
    rallies = []
    current = [ordered[0]]

    for frame in ordered[1:]:
        if frame - current[-1] > max_gap_frames:
            rallies.append(_build_rally(current))
            current = [frame]
        else:
            current.append(frame)
    rallies.append(_build_rally(current))
    return rallies


def _build_rally(bounce_frames):
    return Rally(
        start_frame=bounce_frames[0],
        end_frame=bounce_frames[-1],
        bounce_frames=tuple(bounce_frames),
    )


def propose_outcome(
    rally,
    court_points_m,
    court_type="singles",
    tolerance_m=None,
    position_search_frames=DEFAULT_POSITION_SEARCH_FRAMES,
):
    """Guess who won a rally from its final bounce.

    The reasoning, and its two weak links:

    * A bounce landing in a player's half was hit by their opponent. This is solid.
    * If that final bounce was **out**, the hitter loses. Reasonably solid -- assuming
      bounce detection actually caught the last bounce.
    * If it was **in**, we assume the receiver failed to return it, so the hitter
      wins. This is the weak one: a rally also ends on a net ball, a miss after a good
      bounce, or simply a bounce the detector missed.

    Hence MAX_PROPOSAL_CONFIDENCE. Treat the result as a prompt for a human, not an
    answer.

    :param rally: a Rally from segment_rallies
    :param court_points_m: per-frame ball positions in court meters (index-aligned
        with the original video), as from homography.project_track
    :return: RallyOutcome; .winner is None when no usable bounce position exists
    """
    final_frame, final_point, offset = _last_positioned_bounce(
        rally, court_points_m, position_search_frames
    )
    if final_frame is None:
        return RallyOutcome(
            winner=None,
            reason=(
                "no ball position within "
                f"{position_search_frames} frames of any bounce in this rally"
            ),
            confidence=0.0,
        )

    kwargs = {"tolerance_m": tolerance_m} if tolerance_m is not None else {}
    call = classify_bounce(final_point, shot_type="rally", court_type=court_type, **kwargs)

    offset_penalty = 1.0
    if offset and position_search_frames:
        offset_penalty = 1.0 - MAX_OFFSET_CONFIDENCE_PENALTY * (offset / position_search_frames)

    landed_side = 0 if final_point[1] > rules.NET_Y_M else 1
    hitter = 1 - landed_side

    if call.is_in:
        winner = hitter
        reason = (
            f"final bounce landed in on player {landed_side}'s side; "
            f"assuming player {landed_side} failed to return it"
        )
    else:
        winner = landed_side
        reason = f"final bounce by player {hitter} landed out"
    if offset:
        reason += f" (position taken {offset} frame(s) away; no exact track at the bounce)"

    return RallyOutcome(
        winner=winner,
        reason=reason,
        confidence=min(call.confidence, 1.0) * MAX_PROPOSAL_CONFIDENCE * offset_penalty,
        bounce_frame=final_frame,
        bounce_point_m=final_point,
        was_in=call.is_in,
    )


def _last_positioned_bounce(rally, court_points_m, search_frames):
    """Latest bounce with a usable court position, allowing a small frame search.

    :return: (bounce_frame, point, offset) where offset is how many frames away the
        position was borrowed from (0 = exact); (None, None, None) if nothing usable
    """
    for frame in reversed(rally.bounce_frames):
        found = _nearest_position(court_points_m, frame, search_frames)
        if found is not None:
            offset, point = found
            return frame, point, offset
    return None, None, None


def _nearest_position(court_points_m, frame, search_frames):
    """Closest tracked position to `frame`, within search_frames. (offset, point)."""
    for offset in range(search_frames + 1):
        candidates = (frame,) if offset == 0 else (frame - offset, frame + offset)
        for candidate in candidates:
            if 0 <= candidate < len(court_points_m) and court_points_m[candidate] is not None:
                return offset, court_points_m[candidate]
    return None


def propose_outcomes(rallies, court_points_m, **kwargs):
    """propose_outcome over a list of rallies. Returns [(rally, outcome), ...]."""
    return [(rally, propose_outcome(rally, court_points_m, **kwargs)) for rally in rallies]
