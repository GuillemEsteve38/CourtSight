"""Score a clip from cached perception output, with human-in-the-loop confirmation.

Run scripts/run_perception.py first, then:

    python scripts/score_match.py --input data/samples/xenia_720p.mp4

For each detected rally it shows the proposed winner and the reasoning, and asks you
to confirm. The automation is a prompt, not an authority -- proposals are capped well
below certainty (see src/analysis/rally_segmentation.py) because ball position alone
cannot show a racket miss or a net cord.

Controls: [Enter] accept, [0]/[1] override, [s] skip, [u] undo, [q] stop.
Use --auto to accept every proposal unattended (useful for eyeballing quality, not
for a scoreline you'd trust).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.rally_segmentation import propose_outcome as propose_rally_outcome
from src.analysis.rally_segmentation import segment_rallies
from src.perception import ball_tracking, bounce_detection, court_detection, video_io
from src.perception.homography import project_track
from src.scoring.state_machine import MatchRules, MatchScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="the video passed to run_perception.py")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--court-type", default="singles", choices=["singles", "doubles"])
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--first-server", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max-gap-s", type=float, default=2.5,
                        help="quiet time between bounces that separates two rallies")
    parser.add_argument("--auto", action="store_true", help="accept all proposals unattended")
    args = parser.parse_args()

    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO_ROOT / "data" / "processed" / input_path.stem
    for name in ("court.json", "ball.json", "bounces.json"):
        if not (cache_dir / name).exists():
            print(f"ERROR: {cache_dir / name} not found -- run scripts/run_perception.py first",
                  file=sys.stderr)
            sys.exit(1)

    court = court_detection.load_court_detection(cache_dir / "court.json")
    ball = ball_tracking.load_ball_track(cache_dir / "ball.json")
    bounces = bounce_detection.load_bounces(cache_dir / "bounces.json")
    _, fps = video_io.read_frames(input_path)

    court_points = project_track(ball, court.homography_m)
    rallies = segment_rallies(bounces, fps, max_gap_s=args.max_gap_s)
    print(f"{len(bounces)} bounces -> {len(rallies)} candidate rallies at {fps:.2f} fps\n")

    scorer = MatchScorer(rules=MatchRules(best_of=args.best_of), first_server=args.first_server)

    for number, rally in enumerate(rallies, start=1):
        outcome = propose_rally_outcome(rally, court_points, court_type=args.court_type)
        print(f"--- rally {number}/{len(rallies)}  "
              f"frames {rally.start_frame}-{rally.end_frame} "
              f"({rally.duration_s(fps):.1f}s, {rally.n_bounces} bounces)")
        print(f"    proposal: {_describe(outcome)}")

        if args.auto:
            if outcome.winner is None:
                print("    skipped (inconclusive)\n")
                continue
            scorer.apply_outcome(outcome)
        else:
            if not _confirm(scorer, outcome):
                break
        print(f"    score: {scorer.state}\n")

        if scorer.state.winner is not None:
            print(f"match complete: player {scorer.state.winner} wins")
            break

    _summarize(scorer)


def _describe(outcome):
    if outcome.winner is None:
        return f"inconclusive -- {outcome.reason}"
    return (f"player {outcome.winner} (confidence {outcome.confidence:.2f}) -- {outcome.reason}")


def _confirm(scorer, outcome):
    """Ask the human. Returns False to stop scoring."""
    while True:
        default = "" if outcome.winner is None else f" [Enter=player {outcome.winner}]"
        answer = input(f"    winner?{default} 0/1/s(kip)/u(ndo)/q(uit): ").strip().lower()

        if answer == "":
            if outcome.winner is None:
                print("    inconclusive -- please answer 0 or 1, or skip")
                continue
            scorer.apply_outcome(outcome)
            return True
        if answer in ("0", "1"):
            scorer.apply_outcome(outcome, confirmed_winner=int(answer))
            return True
        if answer == "s":
            return True
        if answer == "u":
            try:
                scorer.undo()
                print(f"    undone -> {scorer.state}")
            except ValueError as e:
                print(f"    {e}")
            continue
        if answer == "q":
            return False
        print("    unrecognised input")


def _summarize(scorer):
    print("\n=== summary ===")
    print(f"final score: {scorer.state}")
    print(f"points scored: {len(scorer.points_played)}")
    rate = scorer.correction_rate
    if rate is not None:
        print(f"human corrections: {rate:.0%} of scored points")


if __name__ == "__main__":
    main()
