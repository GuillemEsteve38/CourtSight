import threading

import pytest
from fastapi.testclient import TestClient

from app.backend import main
from app.backend.jobs import DONE, ERROR, QUEUED, RUNNING, JobStore

# --- JobStore ---------------------------------------------------------------------

def test_job_starts_queued_and_completes():
    store = JobStore()
    job = store.create("clip.mp4", "/tmp/clip.mp4")
    assert job.status == QUEUED

    store.submit(job, lambda j: j.update(result={"ok": True})).result(timeout=5)

    assert job.status == DONE
    assert job.progress == 1.0
    assert job.result == {"ok": True}


def test_worker_failure_is_captured_not_raised():
    store = JobStore()
    job = store.create("clip.mp4", "/tmp/clip.mp4")

    def explode(_):
        raise RuntimeError("model exploded")

    store.submit(job, explode).result(timeout=5)

    assert job.status == ERROR
    assert "model exploded" in job.error
    assert job.to_dict()["status"] == ERROR


def test_one_failing_job_does_not_poison_the_worker():
    store = JobStore()
    bad = store.create("bad.mp4", "/tmp/bad.mp4")
    good = store.create("good.mp4", "/tmp/good.mp4")

    store.submit(bad, lambda _: 1 / 0).result(timeout=5)
    store.submit(good, lambda j: j.update(result={"ok": True})).result(timeout=5)

    assert bad.status == ERROR
    assert good.status == DONE


def test_progress_updates_are_visible_while_running():
    store = JobStore()
    job = store.create("clip.mp4", "/tmp/clip.mp4")
    seen_running = threading.Event()
    release = threading.Event()

    def work(j):
        j.update(progress=0.5, stage="court")
        seen_running.set()
        release.wait(timeout=5)

    future = store.submit(job, work)
    assert seen_running.wait(timeout=5)

    snapshot = job.to_dict()
    assert snapshot["status"] == RUNNING
    assert snapshot["progress"] == 0.5
    assert snapshot["stage"] == "court"

    release.set()
    future.result(timeout=5)


def test_jobs_are_retrievable_by_id():
    store = JobStore()
    job = store.create("clip.mp4", "/tmp/clip.mp4")

    assert store.get(job.id) is job
    assert store.get("nope") is None
    assert job in store.all()


def test_job_ids_are_unique():
    store = JobStore()
    ids = {store.create("c.mp4", "/tmp/c.mp4").id for _ in range(50)}
    assert len(ids) == 50


# --- API ------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "store", JobStore())
    return TestClient(main.app)


@pytest.fixture
def no_op_analysis(monkeypatch):
    """Replace the worker so API tests need no GPU, weights, or real video."""
    monkeypatch.setattr(
        main, "_analyze",
        lambda job: job.update(result={"video": job.filename, "bounces": []}),
    )


def test_health_reports_status(client):
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert "cuda" in body


def test_upload_creates_a_job(client, no_op_analysis):
    response = client.post(
        "/api/jobs", files={"file": ("rally.mp4", b"fake video bytes", "video/mp4")}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "rally.mp4"
    assert body["id"]


def test_uploaded_file_is_written_to_disk(client, no_op_analysis, tmp_path):
    job_id = client.post(
        "/api/jobs", files={"file": ("rally.mp4", b"abc123", "video/mp4")}
    ).json()["id"]

    saved = tmp_path / "uploads" / f"{job_id}.mp4"
    assert saved.exists()
    assert saved.read_bytes() == b"abc123"


def test_unsupported_file_type_is_rejected(client):
    response = client.post(
        "/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"]


def test_job_status_can_be_polled(client, no_op_analysis):
    job_id = client.post(
        "/api/jobs", files={"file": ("rally.mp4", b"x", "video/mp4")}
    ).json()["id"]

    body = _wait_for_done(client, job_id)

    assert body["status"] == "done"
    assert body["progress"] == 1.0


def test_result_is_available_once_complete(client, no_op_analysis):
    job_id = client.post(
        "/api/jobs", files={"file": ("rally.mp4", b"x", "video/mp4")}
    ).json()["id"]
    _wait_for_done(client, job_id)

    result = client.get(f"/api/jobs/{job_id}/result").json()

    assert result["video"] == "rally.mp4"


def test_result_before_completion_returns_conflict(client, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(main, "_analyze", lambda job: release.wait(timeout=5))

    job_id = client.post(
        "/api/jobs", files={"file": ("rally.mp4", b"x", "video/mp4")}
    ).json()["id"]

    response = client.get(f"/api/jobs/{job_id}/result")
    release.set()

    assert response.status_code == 409


def test_failed_job_surfaces_its_error(client, monkeypatch):
    def explode(_):
        raise ValueError("no frames decoded")

    monkeypatch.setattr(main, "_analyze", explode)
    job_id = client.post(
        "/api/jobs", files={"file": ("bad.mp4", b"x", "video/mp4")}
    ).json()["id"]

    body = _wait_for_status(client, job_id, "error")

    assert "no frames decoded" in body["error"]


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/result").status_code == 404
    assert client.get("/api/jobs/deadbeef/video").status_code == 404


def test_missing_annotated_video_returns_404(client, no_op_analysis):
    job_id = client.post(
        "/api/jobs", files={"file": ("rally.mp4", b"x", "video/mp4")}
    ).json()["id"]
    _wait_for_done(client, job_id)

    assert client.get(f"/api/jobs/{job_id}/video").status_code == 404


def test_annotated_video_is_served_when_present(client, monkeypatch, tmp_path):
    video = tmp_path / "annotated.mp4"
    video.write_bytes(b"mp4-bytes")
    monkeypatch.setattr(
        main, "_analyze", lambda job: job.update(result={}, annotated_path=video)
    )

    job_id = client.post(
        "/api/jobs", files={"file": ("rally.mp4", b"x", "video/mp4")}
    ).json()["id"]
    _wait_for_done(client, job_id)

    response = client.get(f"/api/jobs/{job_id}/video")

    assert response.status_code == 200
    assert response.content == b"mp4-bytes"


def test_jobs_can_be_listed(client, no_op_analysis):
    for name in ("a.mp4", "b.mp4"):
        client.post("/api/jobs", files={"file": (name, b"x", "video/mp4")})

    listed = client.get("/api/jobs").json()

    assert {job["filename"] for job in listed} == {"a.mp4", "b.mp4"}


# --- Scoring endpoint ------------------------------------------------------------------
# Thin wrapper over the tested MatchScorer; these check the wiring, not the tennis rules.

def test_score_of_no_points_is_love_all(client):
    body = client.post("/api/score", json={"winners": []}).json()

    assert body["points"] == ["0", "0"]
    assert body["games"] == [0, 0]
    assert body["winner"] is None


def test_score_reflects_confirmed_winners(client):
    body = client.post("/api/score", json={"winners": [0, 0, 0, 0]}).json()

    assert body["games"] == [1, 0]
    assert body["points"] == ["0", "0"]


def test_score_reports_deuce(client):
    body = client.post("/api/score", json={"winners": [0, 1, 0, 1, 0, 1]}).json()

    assert body["is_deuce"] is True
    assert body["points"] == ["40", "40"]


def test_score_is_stateless_so_shortening_the_list_undoes_points(client):
    full = client.post("/api/score", json={"winners": [0, 0, 0]}).json()
    undone = client.post("/api/score", json={"winners": [0, 0]}).json()

    assert full["points"] == ["40", "0"]
    assert undone["points"] == ["30", "0"]


def test_score_stops_applying_points_once_the_match_is_won(client):
    winners = [0] * 200  # far more than needed for a best-of-3

    body = client.post("/api/score", json={"winners": winners}).json()

    assert body["winner"] == 0
    assert body["points_applied"] < len(winners)


def test_score_rejects_invalid_input(client):
    assert client.post("/api/score", json={"winners": [0, 2]}).status_code == 400
    assert client.post("/api/score", json={"winners": [], "first_server": 5}).status_code == 400


def _wait_for_status(client, job_id, target, attempts=100):
    for _ in range(attempts):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] == target:
            return body
    raise AssertionError(f"job never reached {target}: {body}")


def _wait_for_done(client, job_id):
    return _wait_for_status(client, job_id, "done")
