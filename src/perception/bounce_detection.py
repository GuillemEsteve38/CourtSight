"""Bounce-frame detection, wrapping vendor/TennisProject's BounceDetector (CatBoost)."""

from . import _vendor_bridge
from .cache import load_json, save_json

_vendor_bridge.ensure()
from bounce_detector import (
    BounceDetector,
)


def detect_bounces(ball_track, weights_path):
    """Detect bounce frames from a ball pixel track.

    :param ball_track: list of (x, y) or None, as returned by ball_tracking.track_ball
    :param weights_path: path to the pretrained CatBoost bounce_detector weights file
    :return: sorted list of frame indices where the ball bounces
    """
    x_ball = [p[0] if p is not None else None for p in ball_track]
    y_ball = [p[1] if p is not None else None for p in ball_track]

    detector = BounceDetector(str(weights_path))
    bounce_frames = detector.predict(x_ball, y_ball)
    return sorted(int(i) for i in bounce_frames)


def save_bounces(bounce_frames, path):
    save_json(bounce_frames, path)


def load_bounces(path):
    return load_json(path)
