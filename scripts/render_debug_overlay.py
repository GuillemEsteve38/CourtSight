"""Render an annotated debug video directly from src/perception/'s cached JSON output,
without needing the vendor pipeline or a separate manual resize step.

Usage (from the repo root, with .venv active) -- point --input at the SAME video you
passed to scripts/run_perception.py:

    python scripts/run_perception.py --input data/samples/xenia2.mp4
    python scripts/render_debug_overlay.py --input data/samples/xenia2.mp4

The frames are put through the same ensure_target_resolution() step run_perception.py
used, so pixel coordinates line up with what's in the cache regardless of the
original file's resolution.
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.perception import ball_tracking, bounce_detection, court_detection, video_io

KEYPOINT_COLOR = (0, 0, 255)  # red, BGR
BALL_COLOR = (0, 255, 0)  # green
BOUNCE_COLOR = (0, 165, 255)  # orange
BALL_TRACE_LENGTH = 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="the same video passed to run_perception.py")
    parser.add_argument(
        "--cache-dir", default=None,
        help="directory with court.json/ball.json/bounces.json "
        "(default: data/processed/<input_stem>/, matching run_perception.py's default)",
    )
    parser.add_argument("--output", default=None, help="default: data/samples/<input_stem>_debug.mp4")
    args = parser.parse_args()

    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO_ROOT / "data" / "processed" / input_path.stem
    output_path = Path(args.output) if args.output else REPO_ROOT / "data" / "samples" / f"{input_path.stem}_debug.mp4"

    for name in ("court.json", "ball.json", "bounces.json"):
        if not (cache_dir / name).exists():
            print(
                f"ERROR: {cache_dir / name} not found -- run scripts/run_perception.py "
                f"--input {input_path} first",
                file=sys.stderr,
            )
            sys.exit(1)

    court_result = court_detection.load_court_detection(cache_dir / "court.json")
    ball_track = ball_tracking.load_ball_track(cache_dir / "ball.json")
    bounces = set(bounce_detection.load_bounces(cache_dir / "bounces.json"))

    frames, fps = video_io.read_frames(input_path)
    frames = video_io.ensure_target_resolution(frames)
    print(f"read {len(frames)} frames at {fps:.2f} fps from {input_path}")

    if len(frames) != len(court_result):
        print(
            f"WARNING: {len(frames)} frames in {input_path} but cache has "
            f"{len(court_result)} entries -- results will be misaligned past the shorter length",
            file=sys.stderr,
        )

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for i, frame in enumerate(frames):
        img = frame.copy()

        if i < len(court_result) and court_result.keypoints_px[i] is not None:
            for x, y in court_result.keypoints_px[i]:
                cv2.circle(img, (int(x), int(y)), radius=0, color=KEYPOINT_COLOR, thickness=10)

        is_bounce = i in bounces
        for j in range(BALL_TRACE_LENGTH):
            frame_idx = i - j
            if frame_idx < 0 or frame_idx >= len(ball_track) or ball_track[frame_idx] is None:
                continue
            x, y = ball_track[frame_idx]
            color = BOUNCE_COLOR if (j == 0 and is_bounce) else BALL_COLOR
            cv2.circle(img, (int(x), int(y)), radius=5, color=color, thickness=2)

        if is_bounce:
            cv2.putText(
                img, "BOUNCE", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=BOUNCE_COLOR, thickness=2,
            )

        writer.write(img)

    writer.release()
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
