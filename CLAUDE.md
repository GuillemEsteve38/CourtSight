# Tennis Match Analysis App

## Purpose
SwingVision-style app: upload a fixed-camera tennis match video → ball tracking, in/out calls on bounces, per-shot speed estimates, and a running score (human-in-the-loop confirmed). Full scope, non-goals, and phased roadmap live in [docs/development-plan.md](docs/development-plan.md) — read it before starting any phase.

## Current phase
**All six phases (0–6) complete; v1 scope is done.** The pipeline runs end-to-end from CLI (`scripts/`) and browser (`src/pipeline.py` + FastAPI in `app/backend/` + React in `app/frontend/`), validated on real club-court footage. 205 tests, none needing a GPU. See README.md for the overview and SETUP.md → "Running the app" to run it.

**The highest-value outstanding work is accuracy, not features** — see `docs/decisions.md`:
1. Speed is trustworthy on dense tracking (broadcast: 138/137/143 km/h) but not on the amateur clips (213–259 km/h). Fix is per-point uncertainty from the homography Jacobian, so far-court measurements self-report as low-confidence. Do not tune thresholds to hide this.
2. Serves are judged with rally rules; correct service-box selection needs the deuce/ad court from the score.
3. Rally boundaries have never been checked frame-by-frame against video.

Two known accuracy limits are documented in `docs/decisions.md` and should be read before trusting output: **speed** is validated on broadcast footage (138/137/143 km/h) but not on the amateur clips, where perspective amplification near the far baseline inflates it; **rally outcomes** are proposals capped at 0.6 confidence and require human confirmation.

## Compute
This dev machine has a local NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM, driver reports CUDA 13.1) alongside its AMD integrated graphics — GPU inference (Phase 0 pipeline run, Phase 1 perception extraction) runs locally via CUDA-enabled torch/torchvision (cu121 wheels). No separate NVIDIA machine is needed. See `docs/decisions.md` for the correction from the original (incorrect) AMD-only assumption.

## Architecture

| Stage | What it does | Source |
|---|---|---|
| Court detection + homography | 14 court keypoints (+1 center) → pixel↔meters transform | yastrebksv/TennisCourtDetector |
| Ball tracking | Heatmap tracker over consecutive frames (handles blur) | yastrebksv/TrackNet |
| Bounce detection | CatBoost on ball trajectory → bounce frames | yastrebksv/TennisProject |
| In/out + speed | Bounce point through homography vs. line rules; displacement/time | Custom (this repo) |
| Scoring | Rally segmentation + game/set state machine, hybrid confirm | Custom (this repo) |

**Foundation:** `yastrebksv/TennisProject` — combines ball tracking, homography, and bounce detection; the only credible path to in/out. **Borrow** the mini-court/speed-visualization approach from `abdullahtarek/tennis_analysis` for presentation only.

**Data flow:** `video → court keypoints + homography (per frame) → ball pixel positions → ball court-coordinates → bounce frames → {in/out, speed} → rally outcomes → score`

**Caching:** GPU inference (court detection, ball tracking, bounce detection) runs once per clip; results are persisted to `data/` as JSON so every downstream stage (in/out, speed, scoring) reads from cache instead of re-running models.

## Reference repos — what to pull from each
- **yastrebksv/TennisProject** — `main.py` (pipeline entrypoint), `ball_detector.py`, `court_detection_net.py`, `bounce_detector.py`, `homography.py`. Expects 1280×720 input. Pinned deps in its `requirements.txt` are old (torch 1.5.0/torchvision 0.6.0, 2020-era) — do not blindly reuse those pins on a modern CUDA/driver stack; verify compatible versions instead (see `docs/decisions.md`).
- **yastrebksv/TennisCourtDetector** — 14 keypoints + 1 center point, heatmap output refined with classical CV postprocessing (line detection + homography). Input 640×360; trained on 1280×720 source frames. Inference scripts: `infer_in_video.py`, `infer_in_image.py`.
- **yastrebksv/TrackNet** — ball heatmap tracker over consecutive frames, 1280×720 @ 30fps in its training data. Inference: `infer_on_video.py`. TennisProject bundles a compatible version of this already — only go to the standalone repo if TennisProject's copy is insufficient.
- **abdullahtarek/tennis_analysis** — mini-court + speed overlay visualization style only, for Phase 5. Not a source of tracking/detection logic.

## Conventions
- Module boundaries follow `src/perception/`, `src/analysis/`, `src/scoring/`, `src/viz/`, orchestrated by `src/pipeline.py` — see [docs/development-plan.md §4](docs/development-plan.md) for the full repo layout.
- Intermediate results (court keypoints/homography, ball court-coordinates, bounce frames) are cached to `data/` as JSON/parquet — never re-run GPU inference to get data already on disk.
- Deterministic logic (homography math, in/out geometry, scoring state machine) must be unit-tested with `pytest` and must not depend on a GPU or model weights.
- One phase per session, one `phase-N-*` branch, commit per working increment.
- Decisions (court model, speed smoothing, scoring rules, dependency version choices) are logged in `docs/decisions.md` as they're made.
