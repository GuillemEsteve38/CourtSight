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


def detect_court(frames, weights_path, device="cuda"):
    """Run court keypoint detection over a list of video frames.

    :param frames: list of BGR np.ndarray frames (e.g. from video_io.read_frames)
    :param weights_path: path to the pretrained court_detector weights file
    :param device: 'cuda' or 'cpu'
    :return: CourtDetectionResult, one entry per input frame
    """
    detector = CourtDetectorNet(str(weights_path), device)
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


def save_court_detection(result, path):
    save_json(result.to_dict(), path)


def load_court_detection(path):
    return CourtDetectionResult.from_dict(load_json(path))
