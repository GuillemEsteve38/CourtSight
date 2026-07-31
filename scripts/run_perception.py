"""CLI entrypoint for src/perception/: run court/ball/bounce detection on a clip and
cache the results to data/processed/<clip_stem>/.

Usage (from the repo root, with .venv active):

    python scripts/run_perception.py --input data/samples/xenia_720p.mp4

Input video must be exactly 1280x720 -- resize/letterbox first if it isn't (see
SETUP.md and docs/decisions.md for why the vendored models assume this resolution).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.perception import ball_tracking, bounce_detection, court_detection, video_io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="path to a 1280x720 video clip")
    parser.add_argument(
        "--weights-dir", default=str(REPO_ROOT / "data" / "weights"),
        help="directory containing court_detector.pt, tracknet.pt, bounce_detector.cbm",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="where to cache court.json/ball.json/bounces.json "
        "(default: data/processed/<clip_stem>/)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    input_path = Path(args.input)
    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / "processed" / input_path.stem

    frames, fps = video_io.read_frames(input_path)
    print(f"read {len(frames)} frames at {fps:.2f} fps from {input_path}")
    width, height = frames[0].shape[1], frames[0].shape[0]
    if (width, height) != (1280, 720):
        print(
            f"WARNING: input is {width}x{height}, not 1280x720 -- court/ball detection "
            "will likely be wrong or crash. Resize/letterbox first (see SETUP.md).",
            file=sys.stderr,
        )

    print("running court detection...")
    court_result = court_detection.detect_court(frames, weights_dir / "court_detector.pt", device=args.device)
    detected = sum(1 for k in court_result.keypoints_px if k is not None)
    print(f"  keypoints detected on {detected}/{len(court_result)} frames")

    print("running ball tracking...")
    ball_track = ball_tracking.track_ball(frames, weights_dir / "tracknet.pt", device=args.device)
    ball_detected = sum(1 for p in ball_track if p is not None)
    print(f"  ball detected on {ball_detected}/{len(ball_track)} frames")

    print("running bounce detection...")
    bounces = bounce_detection.detect_bounces(ball_track, weights_dir / "bounce_detector.cbm")
    print(f"  bounce frames: {bounces}")

    court_detection.save_court_detection(court_result, out_dir / "court.json")
    ball_tracking.save_ball_track(ball_track, out_dir / "ball.json")
    bounce_detection.save_bounces(bounces, out_dir / "bounces.json")
    print(f"cached results to {out_dir}")


if __name__ == "__main__":
    main()
