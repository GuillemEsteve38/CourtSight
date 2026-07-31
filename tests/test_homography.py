import cv2
import numpy as np
import pytest

from src.perception.homography import (
    MIN_CORRESPONDENCES,
    REFERENCE_KEYPOINTS_M,
    court_to_pixel,
    estimate_homography,
    pixel_to_court,
)


def _synthetic_pixel_points(meters_to_pixel_matrix, dropout_indices=frozenset()):
    """Project REFERENCE_KEYPOINTS_M through a known matrix to fake "detected" pixels."""
    points = []
    for i, m_point in enumerate(REFERENCE_KEYPOINTS_M):
        if i in dropout_indices:
            points.append(None)
            continue
        px, py = pixel_to_court(m_point, meters_to_pixel_matrix)
        points.append((px, py))
    return points


@pytest.fixture
def known_affine_matrix():
    # meters -> pixels: simple scale (100 px/m) + offset, no rotation/perspective.
    return np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 80.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def known_perspective_matrix():
    # A genuine (non-affine) homography: meters -> pixels with perspective foreshortening,
    # built from four correspondences the way a real camera view would produce them.
    meters_corners = np.array(
        [[0.0, 0.0], [10.97, 0.0], [10.97, 23.77], [0.0, 23.77]], dtype=np.float32
    )
    pixel_corners = np.array(
        [[300.0, 100.0], [980.0, 100.0], [1150.0, 700.0], [130.0, 700.0]], dtype=np.float32
    )
    matrix, _ = cv2.findHomography(meters_corners, pixel_corners)
    return matrix


def test_estimate_homography_recovers_affine_transform(known_affine_matrix):
    pixel_points = _synthetic_pixel_points(known_affine_matrix)

    homography = estimate_homography(pixel_points)

    assert homography is not None
    for pixel_point, expected_meters in zip(pixel_points, REFERENCE_KEYPOINTS_M):
        recovered = pixel_to_court(pixel_point, homography)
        assert recovered == pytest.approx(expected_meters, abs=1e-3)


def test_estimate_homography_recovers_perspective_transform(known_perspective_matrix):
    pixel_points = _synthetic_pixel_points(known_perspective_matrix)

    homography = estimate_homography(pixel_points)

    assert homography is not None
    for pixel_point, expected_meters in zip(pixel_points, REFERENCE_KEYPOINTS_M):
        recovered = pixel_to_court(pixel_point, homography)
        assert recovered == pytest.approx(expected_meters, abs=1e-2)


def test_estimate_homography_tolerates_missing_points(known_affine_matrix):
    # Drop all but exactly MIN_CORRESPONDENCES points -- should still recover an exact fit.
    keep = set(range(MIN_CORRESPONDENCES))
    dropout = set(range(len(REFERENCE_KEYPOINTS_M))) - keep
    pixel_points = _synthetic_pixel_points(known_affine_matrix, dropout_indices=dropout)

    homography = estimate_homography(pixel_points)

    assert homography is not None
    for i in keep:
        recovered = pixel_to_court(pixel_points[i], homography)
        assert recovered == pytest.approx(REFERENCE_KEYPOINTS_M[i], abs=1e-3)


def test_estimate_homography_returns_none_below_minimum(known_affine_matrix):
    dropout = set(range(len(REFERENCE_KEYPOINTS_M) - (MIN_CORRESPONDENCES - 1)))
    pixel_points = _synthetic_pixel_points(known_affine_matrix, dropout_indices=dropout)

    assert estimate_homography(pixel_points) is None


def test_estimate_homography_rejects_wrong_length():
    with pytest.raises(ValueError):
        estimate_homography([(0, 0)] * 3)


def test_court_to_pixel_is_inverse_of_pixel_to_court(known_affine_matrix):
    pixel_points = _synthetic_pixel_points(known_affine_matrix)
    homography = estimate_homography(pixel_points)

    for pixel_point in pixel_points:
        meters_point = pixel_to_court(pixel_point, homography)
        roundtrip_pixel = court_to_pixel(meters_point, homography)
        assert roundtrip_pixel == pytest.approx(pixel_point, abs=1e-2)
