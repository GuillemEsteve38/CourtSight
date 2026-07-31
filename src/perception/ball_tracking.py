"""Ball pixel-position tracking, wrapping vendor/TennisProject's BallDetector."""

from . import _vendor_bridge
from .cache import load_json, save_json

_vendor_bridge.ensure()
from ball_detector import BallDetector


def track_ball(frames, weights_path, device="cuda"):
    """Run ball tracking over a list of video frames.

    :param frames: list of BGR np.ndarray frames (e.g. from video_io.read_frames)
    :param weights_path: path to the pretrained tracknet ball-tracking weights file
    :param device: 'cuda' or 'cpu'
    :return: list of (x, y) pixel positions, one per frame; None where undetected
        (the first two frames are always None -- the model needs 2 prior frames of context)
    """
    detector = BallDetector(str(weights_path), device)
    ball_track = detector.infer_model(frames)
    return [(float(x), float(y)) if x is not None else None for x, y in ball_track]


def save_ball_track(ball_track, path):
    save_json(ball_track, path)


def load_ball_track(path):
    return [tuple(p) if p is not None else None for p in load_json(path)]
