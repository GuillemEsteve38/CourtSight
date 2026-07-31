import numpy as np
import pytest

from src.perception import video_io


def _fake_frames(width, height, count=3):
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)]


def test_ensure_target_resolution_noop_when_already_correct():
    frames = _fake_frames(video_io.TARGET_WIDTH, video_io.TARGET_HEIGHT)
    result = video_io.ensure_target_resolution(frames)
    assert result is frames


def test_ensure_target_resolution_resizes_close_to_16_9():
    frames = _fake_frames(848, 478)  # real-world case: 0.2% off from 16:9
    result = video_io.ensure_target_resolution(frames)
    assert len(result) == len(frames)
    for frame in result:
        assert frame.shape[:2] == (video_io.TARGET_HEIGHT, video_io.TARGET_WIDTH)


def test_ensure_target_resolution_rejects_far_from_16_9():
    frames = _fake_frames(1080, 1920)  # portrait, nowhere near 16:9
    with pytest.raises(ValueError):
        video_io.ensure_target_resolution(frames)


def test_ensure_target_resolution_handles_empty_list():
    assert video_io.ensure_target_resolution([]) == []
