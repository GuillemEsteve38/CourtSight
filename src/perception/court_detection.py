"""Court keypoint + homography detection, wrapping vendor/TennisProject's CourtDetectorNet.

Returns, per frame: the 14 court keypoints in video-pixel coordinates, and a
pixel -> court-meters homography (via src/perception/homography.py, not vendor's own
internal reference-image pixel scale -- see docs/decisions.md).
"""

from dataclasses import dataclass

from . import _vendor_bridge
from . import homography as court_homography
from .cache import load_json, save_json

_vendor_bridge.ensure()
from court_detection_net import (
    CourtDetectorNet,
)


@dataclass
class CourtDetectionResult:
    keypoints_px: list  # per frame: list of 14 (x, y) tuples, or None if undetected
    homography_m: list  # per frame: 3x3 pixel->meters matrix (list of lists), or None

    def to_dict(self):
        return {"keypoints_px": self.keypoints_px, "homography_m": self.homography_m}

    @classmethod
    def from_dict(cls, d):
        keypoints_px = [
            [tuple(point) for point in frame_kps] if frame_kps is not None else None
            for frame_kps in d["keypoints_px"]
        ]
        return cls(keypoints_px=keypoints_px, homography_m=d["homography_m"])

    def __len__(self):
        return len(self.keypoints_px)


def detect_court(frames, weights_path, device="cuda", sample_every=1):
    """Run court keypoint detection over a list of video frames.

    :param frames: list of BGR np.ndarray frames (e.g. from video_io.read_frames)
    :param weights_path: path to the pretrained court_detector weights file
    :param device: 'cuda' or 'cpu'
    :param sample_every: run inference on every Nth frame and reuse the nearest
        result for the rest. Only sound for a fixed camera (a v1 assumption), where
        the court does not move; this is the slowest stage, so N=10 cuts total
        analysis time roughly in half.
    :return: CourtDetectionResult, one entry per input frame
    """
    detector = CourtDetectorNet(str(weights_path), device)

    if sample_every > 1:
        sampled_indices = list(range(0, len(frames), sample_every))
        sampled = [frames[i] for i in sampled_indices]
        _, sampled_kps = detector.infer_model(sampled)
        kps_res = _expand_samples(sampled_indices, sampled_kps, len(frames))
    else:
        _, kps_res = detector.infer_model(frames)

    keypoints_px = []
    homography_m = []
    for kps in kps_res:
        if kps is None:
            keypoints_px.append(None)
            homography_m.append(None)
            continue
        points = [(float(p[0, 0]), float(p[0, 1])) for p in kps]
        keypoints_px.append(points)
        matrix = court_homography.estimate_homography(points)
        homography_m.append(matrix.tolist() if matrix is not None else None)

    return CourtDetectionResult(keypoints_px=keypoints_px, homography_m=homography_m)


def _expand_samples(sampled_indices, sampled_values, total):
    """Map results computed on sampled frames back onto every frame index."""
    expanded = []
    for index in range(total):
        position = min(
            range(len(sampled_indices)),
            key=lambda i: abs(sampled_indices[i] - index),
        )
        expanded.append(sampled_values[position])
    return expanded


def consolidate_fixed_camera(result, min_frames=5):
    """Collapse per-frame keypoints into one consolidated set via the median.

    On a fixed camera there is exactly *one* true homography; the per-frame estimates
    are noisy samples of it. Measured on real footage, that noise moves a fixed pixel
    by a standard deviation of ~37 cm in court metres -- seven times the 5 cm
    tolerance the in/out engine uses to decide a line call, and a plausible
    contributor to the spurious speed peaks documented in docs/decisions.md.

    Taking the per-keypoint median across frames averages that noise down and is
    robust to the occasional badly-detected frame, which a mean would not be.

    Only valid when the camera really is fixed (an explicit v1 scope assumption). On
    moving footage this would smear the court into one wrong average position.

    :return: a new CourtDetectionResult with the same consolidated keypoints and
        homography on every frame; returns `result` unchanged if too few frames
        carry keypoints to take a meaningful median
    """
    detected = [kps for kps in result.keypoints_px if kps is not None]
    if len(detected) < min_frames:
        return result

    n_points = len(detected[0])
    consolidated = []
    for point_index in range(n_points):
        xs = sorted(frame[point_index][0] for frame in detected)
        ys = sorted(frame[point_index][1] for frame in detected)
        consolidated.append((_median(xs), _median(ys)))

    matrix = court_homography.estimate_homography(consolidated)
    if matrix is None:
        return result

    n_frames = len(result.keypoints_px)
    return CourtDetectionResult(
        keypoints_px=[list(consolidated)] * n_frames,
        homography_m=[matrix.tolist()] * n_frames,
    )


def _median(sorted_values):
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def save_court_detection(result, path):
    save_json(result.to_dict(), path)


def load_court_detection(path):
    return CourtDetectionResult.from_dict(load_json(path))
