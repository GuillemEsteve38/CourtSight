"""Tennis court reference model in meters (ITF regulation dimensions), for in/out line-calling.

Coordinate convention matches src/perception/homography.py: origin at the top-left
doubles corner, x across the court width, y along the court length (top baseline ->
bottom baseline). Defined independently of homography.py's REFERENCE_KEYPOINTS_M --
that model is a 14 named-point set shaped for homography estimation; this one models
the full regions (rectangles) needed for in/out line-calling. Both derive from the
same public ITF regulation numbers (23.77m x 10.97m doubles court, 8.23m singles
width, 6.40m service line from net), so the duplication can't drift in a way that
matters -- see docs/decisions.md.
"""

from dataclasses import dataclass

DOUBLES_LENGTH_M = 23.77
DOUBLES_WIDTH_M = 10.97
SINGLES_WIDTH_M = 8.23
SERVICE_LINE_FROM_NET_M = 6.40

SINGLES_INSET_M = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2  # 1.37
NET_Y_M = DOUBLES_LENGTH_M / 2  # 11.885
TOP_SERVICE_Y_M = NET_Y_M - SERVICE_LINE_FROM_NET_M  # 5.485
BOTTOM_SERVICE_Y_M = NET_Y_M + SERVICE_LINE_FROM_NET_M  # 18.285
CENTER_X_M = DOUBLES_WIDTH_M / 2  # 5.485
LEFT_SINGLES_X_M = SINGLES_INSET_M
RIGHT_SINGLES_X_M = DOUBLES_WIDTH_M - SINGLES_INSET_M


@dataclass(frozen=True)
class Rect:
    """An axis-aligned court region, in meters."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def signed_distance(self, point):
        """Distance from point to the rect boundary.

        Positive when the point is inside (distance to the nearest edge -- how far
        you could move before leaving the rect). Negative when outside (distance to
        the nearest point on the rect).
        """
        px, py = point
        dx = max(self.x_min - px, 0.0, px - self.x_max)
        dy = max(self.y_min - py, 0.0, py - self.y_max)
        if dx == 0.0 and dy == 0.0:
            return min(px - self.x_min, self.x_max - px, py - self.y_min, self.y_max - py)
        return -((dx**2 + dy**2) ** 0.5)

    def contains(self, point, tolerance=0.0):
        return self.signed_distance(point) >= -tolerance


SINGLES_COURT = Rect(LEFT_SINGLES_X_M, RIGHT_SINGLES_X_M, 0.0, DOUBLES_LENGTH_M)
DOUBLES_COURT = Rect(0.0, DOUBLES_WIDTH_M, 0.0, DOUBLES_LENGTH_M)

# Service boxes are always singles-width, even in doubles play.
SERVICE_BOXES = {
    "top_left": Rect(LEFT_SINGLES_X_M, CENTER_X_M, TOP_SERVICE_Y_M, NET_Y_M),
    "top_right": Rect(CENTER_X_M, RIGHT_SINGLES_X_M, TOP_SERVICE_Y_M, NET_Y_M),
    "bottom_left": Rect(LEFT_SINGLES_X_M, CENTER_X_M, NET_Y_M, BOTTOM_SERVICE_Y_M),
    "bottom_right": Rect(CENTER_X_M, RIGHT_SINGLES_X_M, NET_Y_M, BOTTOM_SERVICE_Y_M),
}
