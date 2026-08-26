"""FastAPI backend: upload a clip, poll an async analysis job, fetch results.

Endpoints
    POST /api/jobs                 upload a video, returns a job id
    GET  /api/jobs                 list jobs
    GET  /api/jobs/{id}            status + progress
    GET  /api/jobs/{id}/result     the AnalysisResult once complete
    GET  /api/jobs/{id}/video      the annotated video (streamed, range-friendly)
    GET  /api/health               liveness + whether CUDA is available

The built frontend, if present at app/frontend/dist, is served from /.

Run it:  uvicorn app.backend.main:app --reload
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.backend.jobs import DONE, JobStore
from src.perception import video_io
from src.pipeline import analyze_video
from src.scoring.state_machine import MatchRules, MatchScorer
from src.viz import overlay

REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = Path(os.environ.get("COURTSIGHT_UPLOAD_DIR", REPO_ROOT / "data" / "uploads"))
WEIGHTS_DIR = Path(os.environ.get("COURTSIGHT_WEIGHTS_DIR", REPO_ROOT / "data" / "weights"))
FRONTEND_DIST = REPO_ROOT / "app" / "frontend" / "dist"

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
# Rendering the annotated video roughly doubles the wall time, so it's the last 25%
# of reported progress rather than a silent extra wait after "analysis complete".
RENDER_PROGRESS_SHARE = 0.25

app = FastAPI(title="CourtSight", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    # The Vite dev server runs on a different origin; the packaged app is same-origin.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = JobStore()


@app.get("/api/health")
def health():
    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else None
    except Exception:  # noqa: BLE001 -- health must never fail on an import problem
        cuda, device = False, None
    return {
        "status": "ok",
        "cuda": cuda,
        "device": device,
        "weights_present": WEIGHTS_DIR.is_dir(),
    }


@app.post("/api/jobs", status_code=201)
def create_job(file: UploadFile = File(...)):
    # Sync on purpose: writing a multi-hundred-MB upload is blocking IO, and inside an
    # async endpoint it would stall the event loop -- freezing the progress polling the
    # UI depends on. A sync endpoint is run in FastAPI's threadpool instead.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {suffix!r}; expected one of {sorted(ALLOWED_SUFFIXES)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job = store.create(filename=file.filename, video_path=UPLOAD_DIR / "pending")
    destination = UPLOAD_DIR / f"{job.id}{suffix}"

    # Streamed to disk rather than read into memory: match videos are large.
    with open(destination, "wb") as out:
        shutil.copyfileobj(file.file, out)
    job.update(video_path=destination)

    store.submit(job, _analyze)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return [job.to_dict() for job in store.all()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _require_job(job_id).to_dict()


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    job = _require_job(job_id)
    if job.status != DONE or job.result is None:
        raise HTTPException(status_code=409, detail=f"job is {job.status}, not ready")
    return job.result


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    job = _require_job(job_id)
    if not job.annotated_path or not Path(job.annotated_path).exists():
        raise HTTPException(status_code=404, detail="annotated video not available")
    return FileResponse(job.annotated_path, media_type="video/mp4")


class ScoreRequest(BaseModel):
    """Confirmed point winners, oldest first."""

    winners: list[int] = Field(default_factory=list)
    best_of: int = 3
    first_server: int = 0


@app.post("/api/score")
def score(request: ScoreRequest):
    """Score a sequence of confirmed rally winners.

    Stateless on purpose: the client owns the list of confirmed winners and re-posts
    it whenever it changes, so undo/override in the UI is just "send a shorter list".
    Scoring itself runs through the tested MatchScorer rather than being reimplemented
    in JavaScript -- there must be exactly one implementation of the tennis rules.
    """
    if any(w not in (0, 1) for w in request.winners):
        raise HTTPException(status_code=400, detail="winners must all be 0 or 1")
    if request.first_server not in (0, 1):
        raise HTTPException(status_code=400, detail="first_server must be 0 or 1")

    scorer = MatchScorer(
        rules=MatchRules(best_of=request.best_of), first_server=request.first_server
    )
    applied = 0
    for winner in request.winners:
        if scorer.state.winner is not None:
            break  # match already decided; ignore any trailing points
        scorer.award_point(winner)
        applied += 1

    state = scorer.state
    return {
        "summary": str(state),
        "points": list(state.point_score_display()),
        "games": list(state.games),
        "sets": list(state.sets),
        "completed_sets": [list(s) for s in state.completed_sets],
        "server": state.server,
        "winner": state.winner,
        "in_tiebreak": state.in_tiebreak,
        "is_deuce": state.is_deuce,
        "points_applied": applied,
    }


def _require_job(job_id):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


def _analyze(job):
    """Worker body: run the pipeline, then render the annotated video."""
    analysis_share = 1.0 - RENDER_PROGRESS_SHARE

    def on_progress(fraction, stage, message):
        job.update(progress=fraction * analysis_share, stage=stage, message=message)

    result = analyze_video(
        job.video_path,
        WEIGHTS_DIR,
        progress=on_progress,
        device=_device(),
    )
    job.update(result=result.to_dict(), stage="render", message="rendering annotated video")

    frames, fps = video_io.read_frames(job.video_path)
    frames = video_io.ensure_target_resolution(frames)
    output = Path(job.video_path).with_name(f"{job.id}_annotated.mp4")

    from src.perception import ball_tracking, court_detection
    from src.pipeline import cache_dir_for

    cache = cache_dir_for(job.video_path)
    court = court_detection.load_court_detection(cache / "court.json")
    ball = ball_tracking.load_ball_track(cache / "ball.json")

    overlay.render_annotated_video(
        frames,
        output,
        fps=fps,
        court_keypoints=court.keypoints_px,
        ball_track=ball,
        court_points_m=[
            None if p is None else tuple(p) for p in result.ball_track_m
        ],
        bounce_calls={b.frame: b.is_in for b in result.bounces},
        progress=lambda f: job.update(
            progress=analysis_share + f * RENDER_PROGRESS_SHARE
        ),
    )
    job.update(annotated_path=output, message="complete")


def _device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


# Mounted last so the API routes above take precedence.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
