"""Pixel <-> court-meters mapping for the 14 court keypoints.

Court model is standard ITF doubles-court regulation dimensions (meters), with the
origin at the top-left doubles corner, x increasing across the court width, y
increasing along the court length (top baseline -> bottom baseline):

    doubles court:  23.77m (length) x 10.97m (width)
    singles court:  8.23m wide, inset 1.37m from each doubles sideline
    service line:   6.40m from the net on each side (net at length / 2 = 11.885m)

Keypoint order and grouping matches vendor/TennisProject/court_reference.py's
`CourtReference.key_points` exactly, since court_detection.py's output (14 detected
keypoints per frame, in video-pixel coordinates) is indexed the same way:

    0: baseline_top left (doubles)       1: baseline_top right (doubles)
    2: baseline_bottom left (doubles)    3: baseline_bottom right (doubles)
    4: left_inner_line top (singles)     5: left_inner_line bottom (singles)
    6: right_inner_line top (singles)    7: right_inner_line bottom (singles)
    8: top_inner_line left               9: top_inner_line right
    10: bottom_inner_line left           11: bottom_inner_line right
    12: middle_line top                  13: middle_line bottom

This module only deals with the geometry: it never touches a GPU or model weights,
so it's fully unit-testable with synthetic point correspondences.
"""

import cv2
import numpy as np

DOUBLES_LENGTH_M = 23.77
DOUBLES_WIDTH_M = 10.97
SINGLES_WIDTH_M = 8.23
SERVICE_LINE_FROM_NET_M = 6.40

_SINGLES_INSET_M = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2  # 1.37
_NET_Y_M = DOUBLES_LENGTH_M / 2  # 11.885
_TOP_SERVICE_Y_M = _NET_Y_M - SERVICE_LINE_FROM_NET_M  # 5.485
_BOTTOM_SERVICE_Y_M = _NET_Y_M + SERVICE_LINE_FROM_NET_M  # 18.285
_CENTER_X_M = DOUBLES_WIDTH_M / 2  # 5.485
_LEFT_SINGLES_X_M = _SINGLES_INSET_M
_RIGHT_SINGLES_X_M = DOUBLES_WIDTH_M - _SINGLES_INSET_M

REFERENCE_KEYPOINTS_M = [
    (0.0, 0.0),                                   # 0 baseline_top left
    (DOUBLES_WIDTH_M, 0.0),                       # 1 baseline_top right
    (0.0, DOUBLES_LENGTH_M),                      # 2 baseline_bottom left
    (DOUBLES_WIDTH_M, DOUBLES_LENGTH_M),           # 3 baseline_bottom right
    (_LEFT_SINGLES_X_M, 0.0),                     # 4 left_inner_line top
    (_LEFT_SINGLES_X_M, DOUBLES_LENGTH_M),          # 5 left_inner_line bottom
    (_RIGHT_SINGLES_X_M, 0.0),                    # 6 right_inner_line top
    (_RIGHT_SINGLES_X_M, DOUBLES_LENGTH_M),         # 7 right_inner_line bottom
    (_LEFT_SINGLES_X_M, _TOP_SERVICE_Y_M),         # 8 top_inner_line left
    (_RIGHT_SINGLES_X_M, _TOP_SERVICE_Y_M),        # 9 top_inner_line right
    (_LEFT_SINGLES_X_M, _BOTTOM_SERVICE_Y_M),      # 10 bottom_inner_line left
    (_RIGHT_SINGLES_X_M, _BOTTOM_SERVICE_Y_M),     # 11 bottom_inner_line right
    (_CENTER_X_M, _TOP_SERVICE_Y_M),               # 12 middle_line top
    (_CENTER_X_M, _BOTTOM_SERVICE_Y_M),            # 13 middle_line bottom
]

MIN_CORRESPONDENCES = 4


def estimate_homography(keypoints_px, ransac_threshold=5.0):
    """Estimate the video-pixel -> court-meters homography from detected keypoints.

    :param keypoints_px: sequence of 14 (x, y) pixel points, matching
        REFERENCE_KEYPOINTS_M order; entries may be None where a keypoint wasn't
        detected in this frame.
    :param ransac_threshold: reprojection error threshold (in meters) passed to
        cv2.findHomography's RANSAC estimator.
    :return: 3x3 homography matrix (np.ndarray) mapping pixel -> meters, or None if
        fewer than MIN_CORRESPONDENCES valid points are available.
    """
    if len(keypoints_px) != len(REFERENCE_KEYPOINTS_M):
        raise ValueError(
            f"expected {len(REFERENCE_KEYPOINTS_M)} keypoints, got {len(keypoints_px)}"
        )

    src, dst = [], []
    for point, reference in zip(keypoints_px, REFERENCE_KEYPOINTS_M):
        if point is not None:
            src.append(point)
            dst.append(reference)

    if len(src) < MIN_CORRESPONDENCES:
        return None

    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    method = cv2.RANSAC if len(src) > MIN_CORRESPONDENCES else 0
    matrix, _ = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_threshold)
    return matrix


def pixel_to_court(point_px, homography_matrix):
    """Map a single (x, y) pixel point to court meters using a pixel->meters homography."""
    pts = np.array([[point_px]], dtype=np.float32)
    result = cv2.perspectiveTransform(pts, homography_matrix)
    return float(result[0, 0, 0]), float(result[0, 0, 1])


def court_to_pixel(point_m, homography_matrix):
    """Map a single (x, y) court-meters point back to video pixels."""
    inverse = np.linalg.inv(homography_matrix)
    return pixel_to_court(point_m, inverse)
