import numpy as np
import pytest

from src.scoring import rules
from src.viz import minicourt, overlay


def blank_frame(width=1280, height=720):
    return np.zeros((height, width, 3), dtype=np.uint8)


# --- minicourt geometry ------------------------------------------------------------

def test_court_corners_map_to_panel_corners():
    top_left = minicourt.court_to_panel((0.0, 0.0))
    bottom_right = minicourt.court_to_panel((rules.DOUBLES_WIDTH_M, rules.DOUBLES_LENGTH_M))

    assert top_left == (minicourt.PANEL_MARGIN, minicourt.PANEL_MARGIN)
    assert bottom_right == (
        minicourt.PANEL_WIDTH - minicourt.PANEL_MARGIN,
        minicourt.PANEL_HEIGHT - minicourt.PANEL_MARGIN,
    )


def test_court_centre_maps_to_panel_centre():
    centre = minicourt.court_to_panel((rules.CENTER_X_M, rules.NET_Y_M))

    assert centre[0] == pytest.approx(minicourt.PANEL_WIDTH / 2, abs=1)
    assert centre[1] == pytest.approx(minicourt.PANEL_HEIGHT / 2, abs=1)


def test_mapping_preserves_left_right_and_near_far_ordering():
    left = minicourt.court_to_panel((1.0, 5.0))
    right = minicourt.court_to_panel((9.0, 5.0))
    near_top = minicourt.court_to_panel((5.0, 2.0))
    near_bottom = minicourt.court_to_panel((5.0, 20.0))

    assert left[0] < right[0]
    assert near_top[1] < near_bottom[1]


def test_draw_court_produces_the_expected_panel_size():
    panel = minicourt.draw_court()

    assert panel.shape == (minicourt.PANEL_HEIGHT, minicourt.PANEL_WIDTH, 3)
    assert panel.any(), "lines should have been drawn"


def test_render_marks_in_and_out_bounces_differently():
    in_point = (rules.CENTER_X_M, rules.NET_Y_M + 4.0)
    out_point = (rules.CENTER_X_M, rules.NET_Y_M + 5.0)

    panel = minicourt.render(bounces=[(in_point, True), (out_point, False)])

    in_px = minicourt.court_to_panel(in_point)
    out_px = minicourt.court_to_panel(out_point)
    assert tuple(panel[in_px[1], in_px[0]]) == minicourt.BOUNCE_IN_COLOR
    assert tuple(panel[out_px[1], out_px[0]]) == minicourt.BOUNCE_OUT_COLOR


def test_render_skips_none_entries():
    panel = minicourt.render(ball_points=[None, (5.0, 5.0), None], bounces=[(None, True)])

    assert panel.shape[:2] == (minicourt.PANEL_HEIGHT, minicourt.PANEL_WIDTH)


def test_paste_places_the_panel_in_the_top_right():
    frame = blank_frame()
    panel = minicourt.draw_court()

    minicourt.paste(frame, panel)

    top_right_region = frame[20:20 + panel.shape[0], -20 - panel.shape[1]:-20]
    assert np.array_equal(top_right_region, panel)


def test_paste_is_skipped_when_the_frame_is_too_small():
    tiny = blank_frame(width=60, height=60)

    result = minicourt.paste(tiny, minicourt.draw_court())

    assert not result.any(), "nothing drawn rather than crashing"


# --- frame annotation ------------------------------------------------------------------

def test_annotate_frame_does_not_mutate_the_input():
    frame = blank_frame()

    overlay.annotate_frame(frame, keypoints_px=[(100, 100)])

    assert not frame.any(), "the original frame must be left untouched"


def test_annotate_frame_draws_keypoints_and_ball():
    annotated = overlay.annotate_frame(
        blank_frame(),
        keypoints_px=[(100, 100)],
        ball_track=[(400, 400)],
        frame_index=0,
    )

    assert annotated[100, 100].any()
    assert annotated[380:420, 380:420].any()


def test_bounce_frames_are_coloured_by_the_call():
    out_frame = overlay.annotate_frame(
        blank_frame(), ball_track=[(400, 400)], frame_index=0, bounce_calls={0: False}
    )
    in_frame = overlay.annotate_frame(
        blank_frame(), ball_track=[(400, 400)], frame_index=0, bounce_calls={0: True}
    )

    assert not np.array_equal(out_frame, in_frame)


def test_ball_trail_draws_previous_frames():
    track = [(100, 100), (200, 200), (300, 300)]

    annotated = overlay.annotate_frame(blank_frame(), ball_track=track, frame_index=2)

    assert annotated[95:105, 95:105].any(), "oldest trail point still drawn"
    assert annotated[295:305, 295:305].any()


def test_annotate_frame_tolerates_missing_data():
    annotated = overlay.annotate_frame(blank_frame(), ball_track=[None, None], frame_index=1)

    assert not annotated.any()


# --- video rendering ----------------------------------------------------------------------

def test_render_annotated_video_writes_a_playable_file(tmp_path):
    import cv2

    frames = [blank_frame(320, 180) for _ in range(10)]
    output = tmp_path / "out.mp4"

    overlay.render_annotated_video(
        frames, output, fps=30.0,
        ball_track=[(160, 90)] * 10,
        bounce_calls={5: True},
    )

    assert output.exists() and output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    finally:
        capture.release()


def test_render_annotated_video_reports_progress(tmp_path):
    seen = []

    overlay.render_annotated_video(
        [blank_frame(320, 180) for _ in range(5)], tmp_path / "p.mp4", fps=30.0,
        progress=seen.append,
    )

    assert seen and seen[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_render_annotated_video_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        overlay.render_annotated_video([], tmp_path / "x.mp4", fps=30.0)
