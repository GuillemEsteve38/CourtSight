"""Single entrypoint that runs every stage and assembles one analysis result.

Split deliberately in two:

* `analyze_video()` does the expensive, impure work -- decode video, run the three
  models, read/write the JSON cache. Needs torch and model weights.
* `build_analysis()` turns already-loaded perception data into the final result. Pure
  (no torch, no GPU, no disk), so the assembly logic is unit-testable and the API
  layer can rebuild a response from cache without touching a model.

Caching follows CLAUDE.md: GPU inference runs once per clip and every downstream stage
reads from `data/processed/<clip>/`. Re-running with the same input is cheap.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.analysis.in_out import classify_bounce
from src.analysis.rally_segmentation import (
    propose_outcome,
    segment_rallies,
)
from src.analysis.speed import (
    peak_speeds_between_bounces,
    reject_implausible_positions,
    smooth_track,
)
from src.perception import ball_tracking, bounce_detection, court_detection, video_io
from src.perception.homography import project_track

# Stage weights for progress reporting, roughly proportional to observed runtime on a
# ~700-frame clip (court detection is the slowest by a wide margin).
STAGE_WEIGHTS = {
    "read": 0.05,
    "court": 0.55,
    "ball": 0.30,
    "bounce": 0.02,
    "analyze": 0.08,
}


@dataclass
class BounceCall:
    frame: int
    point_m: tuple
    is_in: bool
    confidence: float
    distance_m: float


@dataclass
class ShotSpeed:
    start_frame: int
    end_frame: int
    kmh: float
    kmh_low: float
    kmh_high: float


@dataclass
class RallyResult:
    start_frame: int
    end_frame: int
    bounce_frames: list
    proposed_winner: int = None
    proposal_reason: str = ""
    proposal_confidence: float = 0.0


@dataclass
class AnalysisResult:
    video: str
    fps: float
    n_frames: int
    court_type: str
    ball_detected_frames: int = 0
    court_detected_frames: int = 0
    bounces: list = field(default_factory=list)
    shots: list = field(default_factory=list)
    rallies: list = field(default_factory=list)
    ball_track_m: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class _ProgressFrames(list):
    """Frame list that reports progress as the vendored models consume it.

    Neither vendor inference loop accepts a progress callback, and both are slow
    enough (tens of seconds) that stage-start/stage-end reporting leaves a progress
    bar apparently frozen. They do differ in how they read frames --
    `court_detection_net` iterates the list, `ball_detector` indexes it -- so both
    access paths report here. That gets real intra-stage progress without patching
    read-only vendored code.

    Progress is high-watermarked because `ball_detector` looks backwards
    (frames[num-1], frames[num-2]) and a bar that goes backwards looks broken.
    """

    def __init__(self, frames, report, min_step=0.005):
        super().__init__(frames)
        self._report = report
        self._min_step = min_step
        self._high_water = 0.0

    def _tick(self, index):
        if not self:
            return
        fraction = min(max(index / len(self), 0.0), 1.0)
        if fraction >= self._high_water + self._min_step:
            self._high_water = fraction
            self._report(fraction)

    def __iter__(self):
        for index, frame in enumerate(list.__iter__(self)):
            self._tick(index)
            yield frame

    def __getitem__(self, index):
        if isinstance(index, int):
            self._tick(index)
        return list.__getitem__(self, index)


def cache_dir_for(video_path, root=None):
    """Where a clip's cached perception output lives."""
    root = Path(root) if root else Path("data") / "processed"
    return root / Path(video_path).stem


def analyze_video(
    video_path,
    weights_dir,
    cache_dir=None,
    court_type="singles",
    device="cuda",
    force=False,
    progress=None,
):
    """Run the full pipeline for one clip, reusing cached inference where possible.

    :param progress: optional callback(fraction, stage, message) with fraction in
        [0, 1]. Used by the API's background job to drive a progress bar.
    :param force: re-run inference even if a cache exists
    :return: AnalysisResult
    """
    video_path = Path(video_path)
    cache = Path(cache_dir) if cache_dir else cache_dir_for(video_path)
    report = _progress_reporter(progress)

    report("read", 0.0, f"reading {video_path.name}")
    frames, fps = video_io.read_frames(video_path)
    if not frames:
        raise ValueError(f"no frames decoded from {video_path} -- is it a valid video?")
    frames = video_io.ensure_target_resolution(frames)
    n_frames = len(frames)
    report("read", 1.0, f"{n_frames} frames at {fps:.2f} fps")

    weights_dir = Path(weights_dir)
    cached = _load_cache(cache) if not force else None

    if cached is not None:
        court_result, ball_track, bounce_frames = cached
        report("court", 1.0, "loaded court detection from cache")
        report("ball", 1.0, "loaded ball tracking from cache")
        report("bounce", 1.0, "loaded bounces from cache")
    else:
        report("court", 0.0, "detecting court")
        court_result = court_detection.detect_court(
            _ProgressFrames(frames, lambda f: report("court", f, "detecting court")),
            weights_dir / "court_detector.pt",
            device=device,
        )
        report("court", 1.0, "court detected")

        report("ball", 0.0, "tracking ball")
        ball_track = ball_tracking.track_ball(
            _ProgressFrames(frames, lambda f: report("ball", f, "tracking ball")),
            weights_dir / "tracknet.pt",
            device=device,
        )
        report("ball", 1.0, "ball tracked")

        report("bounce", 0.0, "detecting bounces")
        bounce_frames = bounce_detection.detect_bounces(
            ball_track, weights_dir / "bounce_detector.cbm"
        )
        report("bounce", 1.0, f"{len(bounce_frames)} bounces")

        court_detection.save_court_detection(court_result, cache / "court.json")
        ball_tracking.save_ball_track(ball_track, cache / "ball.json")
        bounce_detection.save_bounces(bounce_frames, cache / "bounces.json")

    report("analyze", 0.0, "computing calls, speeds and rallies")
    result = build_analysis(
        video=video_path.name,
        fps=fps,
        n_frames=n_frames,
        ball_track=ball_track,
        court_result=court_result,
        bounce_frames=bounce_frames,
        court_type=court_type,
    )
    report("analyze", 1.0, "done")
    return result


def load_cached_analysis(video_path, cache_dir=None, court_type="singles", fps=None, n_frames=None):
    """Rebuild an AnalysisResult purely from cache, without touching a model.

    Returns None when the clip has not been analysed yet.
    """
    cache = Path(cache_dir) if cache_dir else cache_dir_for(video_path)
    cached = _load_cache(cache)
    if cached is None:
        return None
    court_result, ball_track, bounce_frames = cached
    return build_analysis(
        video=Path(video_path).name,
        fps=fps if fps is not None else 30.0,
        n_frames=n_frames if n_frames is not None else len(ball_track),
        ball_track=ball_track,
        court_result=court_result,
        bounce_frames=bounce_frames,
        court_type=court_type,
    )


def build_analysis(
    *,
    video,
    fps,
    n_frames,
    ball_track,
    court_result,
    bounce_frames,
    court_type="singles",
):
    """Assemble the final result from perception output. Pure -- no IO, no models."""
    raw_court_points = project_track(ball_track, court_result.homography_m)
    court_points = reject_implausible_positions(raw_court_points)
    smoothed = smooth_track(court_points, window=5)

    bounces = []
    for frame in bounce_frames:
        point = _position_near(court_points, frame)
        if point is None:
            continue
        call = classify_bounce(point, shot_type="rally", court_type=court_type)
        bounces.append(
            BounceCall(
                frame=frame,
                point_m=(round(point[0], 3), round(point[1], 3)),
                is_in=call.is_in,
                confidence=round(call.confidence, 3),
                distance_m=round(call.distance_m, 3),
            )
        )

    shots = [
        ShotSpeed(
            start_frame=start,
            end_frame=end,
            kmh=round(estimate.kmh, 1),
            kmh_low=round(estimate.kmh_low, 1),
            kmh_high=round(estimate.kmh_high, 1),
        )
        for start, end, estimate in peak_speeds_between_bounces(smoothed, fps, bounce_frames)
    ]

    rallies = []
    for rally in segment_rallies(bounce_frames, fps):
        outcome = propose_outcome(rally, court_points, court_type=court_type)
        rallies.append(
            RallyResult(
                start_frame=rally.start_frame,
                end_frame=rally.end_frame,
                bounce_frames=list(rally.bounce_frames),
                proposed_winner=outcome.winner,
                proposal_reason=outcome.reason,
                proposal_confidence=round(outcome.confidence, 3),
            )
        )

    return AnalysisResult(
        video=video,
        fps=fps,
        n_frames=n_frames,
        court_type=court_type,
        ball_detected_frames=sum(1 for p in ball_track if p is not None),
        court_detected_frames=sum(1 for k in court_result.keypoints_px if k is not None),
        bounces=bounces,
        shots=shots,
        rallies=rallies,
        ball_track_m=[
            None if p is None else [round(p[0], 3), round(p[1], 3)] for p in court_points
        ],
    )


def _position_near(court_points, frame, search=3):
    """Ball position at `frame`, or the nearest within `search` frames.

    Bounce frames don't always have a tracked position -- see docs/decisions.md.
    """
    for offset in range(search + 1):
        for candidate in ((frame,) if offset == 0 else (frame - offset, frame + offset)):
            if 0 <= candidate < len(court_points) and court_points[candidate] is not None:
                return court_points[candidate]
    return None


def _load_cache(cache):
    names = ("court.json", "ball.json", "bounces.json")
    if not all((cache / name).exists() for name in names):
        return None
    return (
        court_detection.load_court_detection(cache / "court.json"),
        ball_tracking.load_ball_track(cache / "ball.json"),
        bounce_detection.load_bounces(cache / "bounces.json"),
    )


def _progress_reporter(callback):
    """Map per-stage progress onto one overall 0..1 fraction."""
    order = list(STAGE_WEIGHTS)

    def report(stage, stage_fraction, message=""):
        if callback is None:
            return
        completed = sum(STAGE_WEIGHTS[s] for s in order[: order.index(stage)])
        overall = completed + STAGE_WEIGHTS[stage] * stage_fraction
        callback(min(overall, 1.0), stage, message)

    return report
