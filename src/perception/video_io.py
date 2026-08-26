"""Video frame reading, shared by the perception wrapper modules.

**This module holds every decoded frame in RAM.** One 1280x720 BGR frame is 2.64 MB,
so a clip costs roughly `frames x 2.64 MB`: ~1.8 GB for 20 seconds, ~4.6 GB per minute,
~23 GB for five minutes. That is a hard ceiling on clip length, not a tuning knob --
the vendored models take a frame *list*, not a stream, so streaming would mean
reimplementing their inference loops.

Two consequences are handled here rather than left to fail as an OOM crash:
`read_frames` refuses clips above a frame budget up front, and
`ensure_target_resolution` resizes in place so the original and resized copies never
coexist. See docs/decisions.md.
"""

import os

import cv2

# vendor/TennisProject's models hardcode assumptions tied to this exact resolution
# (see docs/decisions.md: court_detection_net.py's scale=2 reprojection breaks on
# anything else, both in accuracy and with an outright crash on some inputs).
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
_TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT
_ASPECT_TOLERANCE = 0.02  # 2% -- beyond this, a plain resize would visibly distort the court

BYTES_PER_TARGET_FRAME = TARGET_WIDTH * TARGET_HEIGHT * 3

# ~6.3 GB of frames (2400 x 2.64 MB), which leaves headroom for torch/CUDA and the OS
# on a 16 GB machine. At 30 fps that is 80 seconds of footage. Raise it via
# COURTSIGHT_MAX_FRAMES if you have the RAM -- the failure mode above this is an OOM
# kill or a swapping machine, so the default is deliberately conservative.
DEFAULT_MAX_FRAMES = int(os.environ.get("COURTSIGHT_MAX_FRAMES", "2400"))


def estimated_memory_gb(n_frames):
    """RAM the decoded frames alone will occupy, at target resolution."""
    return n_frames * BYTES_PER_TARGET_FRAME / 1024**3


def read_frames(path_video, max_frames=DEFAULT_MAX_FRAMES):
    """Read every frame of a video into memory.

    :param max_frames: refuse clips longer than this rather than exhausting RAM.
        None disables the check.
    :raises ValueError: if the clip is unreadable, empty, or too long
    :return: (frames, fps) where frames is a list of BGR np.ndarray frames
    """
    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"could not open {path_video} -- unsupported or corrupt video file")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Checked before decoding so an over-long clip fails in milliseconds instead of
        # after filling memory. Container metadata can lie, hence the check while
        # reading too.
        if max_frames and declared > max_frames:
            raise ValueError(_too_long_message(declared, fps, max_frames))

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames and len(frames) > max_frames:
                raise ValueError(_too_long_message(len(frames), fps, max_frames))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"no frames decoded from {path_video} -- is it a valid video?")
    return frames, fps


def _too_long_message(n_frames, fps, max_frames):
    seconds = n_frames / fps if fps else 0
    return (
        f"clip is {n_frames} frames (~{seconds:.0f}s) which needs about "
        f"{estimated_memory_gb(n_frames):.1f} GB of RAM to decode; the limit is "
        f"{max_frames} frames (~{estimated_memory_gb(max_frames):.1f} GB). "
        "Trim the clip, or raise COURTSIGHT_MAX_FRAMES if you have the memory."
    )


def ensure_target_resolution(frames):
    """Resize frames to 1280x720 if they aren't already.

    A plain resize (no letterboxing) is used, which is only safe when the source
    aspect ratio is close to 16:9 -- raises ValueError otherwise rather than silently
    producing a distorted court that would throw off every downstream
    homography/keypoint computation.

    **Mutates `frames` in place** and returns it. Building a second list would mean
    the original and resized frames coexist, roughly doubling peak memory at exactly
    the point where it is already the binding constraint (a 20s clip peaked at ~2.6 GB
    that way). Callers already reassign the return value, so the aliasing is
    invisible to them -- but it is a real side effect, hence this note and a test
    pinning the behaviour.

    :return: the same list, resized
    """
    if not frames:
        return frames

    height, width = frames[0].shape[:2]
    if (width, height) == (TARGET_WIDTH, TARGET_HEIGHT):
        return frames

    aspect = width / height
    if abs(aspect - _TARGET_ASPECT) / _TARGET_ASPECT > _ASPECT_TOLERANCE:
        raise ValueError(
            f"input is {width}x{height} (aspect {aspect:.3f}), too different from "
            f"16:9 ({_TARGET_ASPECT:.3f}) for a plain resize to be safe -- letterbox "
            "it to 1280x720 manually first (see SETUP.md)"
        )

    for index in range(len(frames)):
        frames[index] = cv2.resize(
            frames[index], (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA
        )
    return frames
