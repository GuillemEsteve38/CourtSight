"""Draw perception output onto video frames and write an annotated clip.

Reads only from cached/derived data, never from a model, so re-rendering a clip is
cheap and needs no GPU.
"""

import cv2

from src.viz import minicourt

KEYPOINT_COLOR = (0, 0, 255)
BALL_COLOR = (0, 255, 0)
BOUNCE_IN_COLOR = (0, 200, 255)
BOUNCE_OUT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

BALL_TRAIL_FRAMES = 7


def annotate_frame(
    frame,
    keypoints_px=None,
    ball_track=None,
    frame_index=0,
    bounce_calls=None,
    panel=None,
    caption=None,
):
    """Return a copy of `frame` with court keypoints, ball trail and markers drawn.

    :param bounce_calls: {frame_index: is_in} so a bounce can be coloured by its call
    """
    image = frame.copy()

    for point in keypoints_px or ():
        cv2.circle(image, (int(point[0]), int(point[1])), 0, KEYPOINT_COLOR, 10)

    bounce_calls = bounce_calls or {}
    if ball_track:
        for back in range(BALL_TRAIL_FRAMES):
            index = frame_index - back
            if index < 0 or index >= len(ball_track) or ball_track[index] is None:
                continue
            x, y = ball_track[index]
            if back == 0 and index in bounce_calls:
                color = BOUNCE_IN_COLOR if bounce_calls[index] else BOUNCE_OUT_COLOR
                radius = 8
            else:
                color = BALL_COLOR
                radius = 5
            cv2.circle(image, (int(x), int(y)), radius, color, 2)

    if frame_index in bounce_calls:
        verdict = "IN" if bounce_calls[frame_index] else "OUT"
        color = BOUNCE_IN_COLOR if bounce_calls[frame_index] else BOUNCE_OUT_COLOR
        cv2.putText(image, f"BOUNCE {verdict}", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if caption:
        cv2.putText(image, caption, (20, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    if panel is not None:
        minicourt.paste(image, panel)
    return image


def render_annotated_video(
    frames,
    output_path,
    fps,
    court_keypoints=None,
    ball_track=None,
    court_points_m=None,
    bounce_calls=None,
    captions=None,
    progress=None,
):
    """Write an annotated video. Returns the output path.

    :param court_points_m: ball positions in court meters, used for the mini-court
    :param bounce_calls: {frame_index: is_in}
    :param captions: optional {frame_index: text} burned into the frame
    """
    if not frames:
        raise ValueError("no frames to render")

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {output_path}")

    bounce_calls = bounce_calls or {}
    captions = captions or {}
    seen_bounces = []

    try:
        for index, frame in enumerate(frames):
            if court_points_m is not None:
                if index in bounce_calls and _at(court_points_m, index) is not None:
                    seen_bounces.append((_at(court_points_m, index), bounce_calls[index]))
                panel = minicourt.render(
                    ball_points=court_points_m[max(0, index - 60): index + 1],
                    bounces=seen_bounces,
                )
            else:
                panel = None

            writer.write(
                annotate_frame(
                    frame,
                    keypoints_px=_at(court_keypoints, index),
                    ball_track=ball_track,
                    frame_index=index,
                    bounce_calls=bounce_calls,
                    panel=panel,
                    caption=captions.get(index),
                )
            )
            if progress and index % 25 == 0:
                progress((index + 1) / len(frames))
    finally:
        writer.release()

    if progress:
        progress(1.0)
    return output_path


def _at(sequence, index):
    if sequence is None or index >= len(sequence):
        return None
    return sequence[index]
