import cv2
import numpy as np
import pytest

from src.perception import video_io


def _fake_frames(width, height, count=3):
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)]


def _write_video(path, n_frames, width=1280, height=720, fps=30.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


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


def test_ensure_target_resolution_resizes_in_place():
    """Pinned deliberately: building a second list doubles peak RAM (see the module
    docstring), which is the binding constraint on clip length."""
    frames = _fake_frames(848, 478, count=3)

    result = video_io.ensure_target_resolution(frames)

    assert result is frames, "must not allocate a second list"
    assert frames[0].shape[:2] == (video_io.TARGET_HEIGHT, video_io.TARGET_WIDTH)


# --- Reading real files, and refusing the ones we cannot handle ----------------------

def test_read_frames_reads_a_real_video(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", n_frames=12, width=320, height=180)

    frames, fps = video_io.read_frames(path)

    assert len(frames) == 12
    assert fps == pytest.approx(30.0, abs=0.5)


def test_read_frames_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="could not open"):
        video_io.read_frames(tmp_path / "nope.mp4")


def test_read_frames_rejects_a_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is definitely not a video")

    with pytest.raises(ValueError):
        video_io.read_frames(corrupt)


def test_read_frames_refuses_clips_over_the_frame_budget(tmp_path):
    path = _write_video(tmp_path / "long.mp4", n_frames=40, width=320, height=180)

    with pytest.raises(ValueError) as excinfo:
        video_io.read_frames(path, max_frames=10)

    message = str(excinfo.value)
    assert "GB of RAM" in message, "the error must explain why, not just refuse"
    assert "COURTSIGHT_MAX_FRAMES" in message, "and say how to override it"


def test_read_frames_budget_can_be_disabled(tmp_path):
    path = _write_video(tmp_path / "long.mp4", n_frames=40, width=320, height=180)

    frames, _ = video_io.read_frames(path, max_frames=None)

    assert len(frames) == 40


def test_estimated_memory_matches_frame_size():
    one_frame_gb = video_io.BYTES_PER_TARGET_FRAME / 1024**3

    assert video_io.estimated_memory_gb(1) == pytest.approx(one_frame_gb)
    assert video_io.estimated_memory_gb(1000) == pytest.approx(one_frame_gb * 1000)
    # The number that motivated the budget: a minute at 30fps is multiple GB.
    assert video_io.estimated_memory_gb(1800) > 4.0
