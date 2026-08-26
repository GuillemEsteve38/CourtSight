"""Top-down mini-court panel: a schematic court with ball and bounce positions on it.

Presentation only -- borrowed in spirit from abdullahtarek/tennis_analysis (see
CLAUDE.md), which is a reference for visualisation style, not for detection logic.
Everything drawn here comes from court-meters coordinates, so this panel is also a
useful sanity check: if the mini-court looks wrong, the homography is wrong.
"""

import cv2
import numpy as np

from src.scoring import rules

PANEL_WIDTH = 180
PANEL_HEIGHT = 380
PANEL_MARGIN = 14

BACKGROUND = (35, 35, 35)
LINE_COLOR = (235, 235, 235)
BALL_COLOR = (0, 255, 0)
BOUNCE_IN_COLOR = (0, 200, 255)
BOUNCE_OUT_COLOR = (0, 0, 255)


def court_to_panel(point_m, width=PANEL_WIDTH, height=PANEL_HEIGHT, margin=PANEL_MARGIN):
    """Map a court-meters point onto mini-court panel pixels."""
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    x = margin + (point_m[0] / rules.DOUBLES_WIDTH_M) * usable_w
    y = margin + (point_m[1] / rules.DOUBLES_LENGTH_M) * usable_h
    return round(x), round(y)


def draw_court(width=PANEL_WIDTH, height=PANEL_HEIGHT, margin=PANEL_MARGIN):
    """A blank mini-court with regulation line markings."""
    panel = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    def line(start_m, end_m, thickness=1):
        cv2.line(
            panel,
            court_to_panel(start_m, width, height, margin),
            court_to_panel(end_m, width, height, margin),
            LINE_COLOR,
            thickness,
        )

    w, length = rules.DOUBLES_WIDTH_M, rules.DOUBLES_LENGTH_M
    # Doubles boundary.
    line((0, 0), (w, 0))
    line((0, length), (w, length))
    line((0, 0), (0, length))
    line((w, 0), (w, length))
    # Singles sidelines.
    line((rules.LEFT_SINGLES_X_M, 0), (rules.LEFT_SINGLES_X_M, length))
    line((rules.RIGHT_SINGLES_X_M, 0), (rules.RIGHT_SINGLES_X_M, length))
    # Service lines and centre service line.
    line((rules.LEFT_SINGLES_X_M, rules.TOP_SERVICE_Y_M), (rules.RIGHT_SINGLES_X_M, rules.TOP_SERVICE_Y_M))
    line((rules.LEFT_SINGLES_X_M, rules.BOTTOM_SERVICE_Y_M), (rules.RIGHT_SINGLES_X_M, rules.BOTTOM_SERVICE_Y_M))
    line((rules.CENTER_X_M, rules.TOP_SERVICE_Y_M), (rules.CENTER_X_M, rules.BOTTOM_SERVICE_Y_M))
    # Net.
    line((0, rules.NET_Y_M), (w, rules.NET_Y_M), thickness=2)
    return panel


def render(ball_points=None, bounces=None, width=PANEL_WIDTH, height=PANEL_HEIGHT,
           margin=PANEL_MARGIN, trail_color=BALL_COLOR):
    """Mini-court with an optional ball trail and bounce markers.

    :param ball_points: court-meters positions to draw as a faint trail (None entries
        are skipped)
    :param bounces: iterable of (point_m, is_in); coloured by the call so an out-call
        is visible at a glance
    """
    panel = draw_court(width, height, margin)

    for point in ball_points or ():
        if point is None:
            continue
        cv2.circle(panel, court_to_panel(point, width, height, margin), 1, trail_color, -1)

    for point, is_in in bounces or ():
        if point is None:
            continue
        color = BOUNCE_IN_COLOR if is_in else BOUNCE_OUT_COLOR
        cv2.circle(panel, court_to_panel(point, width, height, margin), 3, color, -1)

    return panel


def paste(frame, panel, margin=20):
    """Composite a mini-court panel into a frame's top-right corner, in place."""
    height, width = panel.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    if height + 2 * margin > frame_h or width + 2 * margin > frame_w:
        return frame  # frame too small to host the panel; skip rather than crash

    top, left = margin, frame_w - margin - width
    frame[top:top + height, left:left + width] = panel
    return frame
