"""In-process job registry for video analysis.

Analysis takes tens of seconds to minutes, far too long for a request/response cycle,
so uploads return a job id immediately and the client polls for progress.

Deliberately simple: an in-memory dict plus a single-worker thread pool. One worker
because the GPU can only usefully run one clip at a time anyway, and serialising the
work keeps VRAM use predictable. A thread (not a process) is fine because the heavy
stages are torch/OpenCV calls that release the GIL.

The consequence -- jobs are lost on restart -- is acceptable for a local tool and is
the honest trade for not dragging in a broker. Swapping this for Celery/RQ later means
reimplementing `submit` and `get`, nothing else.
"""

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"


@dataclass
class Job:
    id: str
    filename: str
    status: str = QUEUED
    progress: float = 0.0
    stage: str = ""
    message: str = "queued"
    error: str = None
    result: dict = None
    video_path: Path = None
    annotated_path: Path = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "filename": self.filename,
                "status": self.status,
                "progress": round(self.progress, 4),
                "stage": self.stage,
                "message": self.message,
                "error": self.error,
                "has_video": self.annotated_path is not None,
            }

    def update(self, **fields):
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)


class JobStore:
    """Thread-safe registry of jobs plus the worker pool that runs them."""

    def __init__(self, max_workers=1):
        self._jobs = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")

    def create(self, filename, video_path):
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, video_path=Path(video_path))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def all(self):
        with self._lock:
            return list(self._jobs.values())

    def submit(self, job, work):
        """Run `work(job)` on the pool, tracking status and surfacing failures.

        `work` receives the Job so it can report progress via job.update().
        """

        def runner():
            job.update(status=RUNNING, message="starting", progress=0.0)
            try:
                work(job)
            except Exception as exc:  # noqa: BLE001 -- must not kill the worker thread
                job.update(
                    status=ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                    message="failed",
                )
                traceback.print_exc()
            else:
                if job.status != ERROR:
                    job.update(status=DONE, progress=1.0, message="complete")

        return self._pool.submit(runner)

    def shutdown(self, wait=True):
        self._pool.shutdown(wait=wait)
