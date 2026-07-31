"""Video frame reading, shared by the perception wrapper modules."""

import cv2

# vendor/TennisProject's models hardcode assumptions tied to this exact resolution
# (see docs/decisions.md: court_detection_net.py's scale=2 reprojection breaks on
# anything else, both in accuracy and with an outright crash on some inputs).
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
_TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT
_ASPECT_TOLERANCE = 0.02  # 2% -- beyond this, a plain resize would visibly distort the court


def read_frames(path_video):
    """Read every frame of a video into memory.

    :return: (frames, fps) where frames is a list of BGR np.ndarray frames.
    """
    cap = cv2.VideoCapture(str(path_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def ensure_target_resolution(frames):
    """Resize frames to 1280x720 if they aren't already.

    A plain resize (no letterboxing) is used, which is only safe when the source
    aspect ratio is close to 16:9 -- raises ValueError otherwise rather than
    silently producing a distorted court that would throw off every downstream
    homography/keypoint computation.

    :return: frames, unchanged if already 1280x720
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

    return [
        cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
