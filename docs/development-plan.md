# Tennis Match Analysis App — Development Plan

A build plan for a SwingVision-style app: upload a match video → ball tracking, in/out calls, ball speed, and live scoring. Written to be executed **in Claude Code**, one scoped session at a time.

---

## 1. Scope

**v1 goal:** From a single fixed-camera video of a rally-based match, produce ball trajectory on a court map, in/out calls on bounces, per-shot speed estimates, and a running score (with human-in-the-loop confirmation).

**In scope for v1**
- Fixed camera, behind the baseline, one court at a time
- Ball tracking + court homography + bounce detection
- In/out from bounce location vs. line geometry
- Approximate ball speed
- Hybrid scoring: auto-propose rally outcome, user confirms

**Explicit non-goals for v1** (park these to avoid scope creep)
- Moving/broadcast multi-camera angles
- True 3D speed and net-clearance (monocular limit — see §8)
- Fully automatic, unattended scoring
- Spin, shot classification, player biomechanics

**Guiding principle:** get someone else's pretrained pipeline running *first*, prove it on your own footage, *then* build the parts that are yours (in/out engine, scoring).

---

## 2. Architecture

Five stages. Four have open-source starting points; scoring is custom.

| Stage | What it does | Source |
|---|---|---|
| Court detection + homography | 14 court keypoints → pixel↔meters transform | yastrebksv/TennisCourtDetector |
| Ball tracking | Heatmap tracker over 3 consecutive frames (handles blur) | yastrebksv/TrackNet |
| Bounce detection | CatBoost on ball trajectory → bounce frames | yastrebksv/TennisProject |
| In/out + speed | Bounce point through homography vs. line rules; displacement/time | **Custom (yours)** |
| Scoring | Rally segmentation + game/set state machine, hybrid confirm | **Custom (yours)** |

**Foundation:** `yastrebksv/TennisProject` (best ball tracking + real homography + bounce detection = the only credible path to in/out). **Borrow** the mini-court and speed-visualization approach from `abdullahtarek/tennis_analysis` for the presentation layer — cleaner and easier to read.

**Data flow:**
`video → court keypoints + homography (per frame) → ball pixel positions → ball court-coordinates → bounce frames → {in/out, speed} → rally outcomes → score`

Persist the intermediate results (court coords + bounces as JSON/parquet) so the expensive GPU inference runs **once** and every downstream stage reads from cache.

---

## 3. Tech stack

- **Language:** Python 3.10+
- **ML:** PyTorch (TrackNet + court detector), CatBoost (bounce)
- **CV/geometry:** OpenCV (video I/O, homography, warping), NumPy
- **Compute:** GPU required for inference. Local NVIDIA GPU if you have one, else Google Colab for the inference stages.
- **Scoring/logic:** plain Python + a tested state machine
- **App layer (Phase 5):** FastAPI backend (async processing job) + React frontend
- **Tooling:** git (branch per phase), pytest, ruff/black, a `notebooks/` folder for experiments

---

## 4. Repo structure

```
tennis-analysis/
├── CLAUDE.md                # project context for Claude Code (see §5)
├── README.md
├── requirements.txt
├── docs/
│   ├── development-plan.md  # this file
│   └── decisions.md         # running log of choices + why
├── data/
│   ├── raw/                 # input videos
│   ├── samples/             # short test clips
│   └── weights/             # pretrained model weights
├── src/
│   ├── perception/
│   │   ├── court_detection.py
│   │   ├── ball_tracking.py
│   │   ├── bounce_detection.py
│   │   └── homography.py
│   ├── analysis/
│   │   ├── in_out.py
│   │   ├── speed.py
│   │   └── rally_segmentation.py
│   ├── scoring/
│   │   ├── state_machine.py
│   │   └── rules.py
│   ├── viz/
│   │   ├── minicourt.py
│   │   └── overlay.py
│   └── pipeline.py          # orchestrates stages, handles caching
├── app/                     # Phase 5
│   ├── backend/
│   └── frontend/
├── tests/
└── notebooks/
```

---

## 5. How to work with Claude Code on this

The single most important setup step: a **`CLAUDE.md`** at the repo root that Claude Code reads every session. Keep it short and current:

- One-paragraph project purpose and the v1 scope from §1
- The architecture table from §2 and the data flow
- Which reference repos are the foundation and what to borrow from each
- Conventions: module boundaries, where intermediate results are cached, test expectations
- A **"current phase / status"** line you update as you go, so each session starts oriented

**Session discipline**
- One phase task per session; commit per working increment on a `phase-N-*` branch.
- Have Claude Code **fetch and read the reference repo code** before writing adapters — don't let it reimplement TrackNet from memory. Point it at the actual files.
- Split the work by who runs what: **Claude Code writes code, scripts, and tests in your environment; you run the GPU-heavy inference** (locally or on Colab) and paste back results/errors.
- Write **tests for the deterministic parts first** — homography math, in/out geometry, and the scoring state machine are all pure logic and fully testable without a GPU. This is where Claude Code shines and where bugs are cheapest to catch.
- Keep a `docs/decisions.md` log so choices (court model, speed smoothing, scoring rules) survive across sessions.

Each phase below includes a **paste-ready starting prompt** for Claude Code.

---

## 6. Phased roadmap

### Phase 0 — Environment & baseline validation
**Goal:** the pretrained `TennisProject` pipeline runs end-to-end on a sample clip and produces annotated output. Nothing custom yet.

**Tasks**
- Scaffold the repo (§4), git init, Python env, `requirements.txt`.
- Decide compute (local GPU vs Colab) and document it in `CLAUDE.md`.
- Fetch the READMEs for `TennisProject`, `TennisCourtDetector`, `TrackNet`; derive exact setup + weight-download steps.
- Download pretrained weights into `data/weights/`.
- Run the pipeline on a provided sample video; verify annotated output + ball trajectory + court map.

**Done when:** a sample clip produces an annotated video and a ball-trajectory/court-map output, reproducibly.

**Gotchas:** CUDA/driver setup; Google-Drive weight links; TennisProject expects **1280×720** input.

> **Claude Code prompt:** "Fetch the READMEs and inference scripts from github.com/yastrebksv/TennisProject, /TennisCourtDetector, and /TrackNet. Scaffold the repo per docs/development-plan.md §4, write requirements.txt, and produce a step-by-step SETUP.md with the exact commands to install deps, download the pretrained weights, and run the pipeline on data/samples/. Flag anything that needs a GPU so I can run those steps myself."

---

### Phase 1 — Perception on your own footage
**Goal:** clean ball trajectory in **court coordinates** + bounce frames, on *your* footage.

**Tasks**
- Capture proper footage: fixed tripod, behind baseline, **≥60 fps**, good even light. (This gates everything downstream — do it well.)
- Adapt input handling to your resolution/format (resize/letterbox to 1280×720 as needed).
- Extract per frame: ball pixel position, court keypoints, homography matrix; plus bounce frames.
- Wrap it behind a clean module API in `src/perception/` and **cache intermediate results** to disk.

**Done when:** for one rally clip you have a structured file of ball court-coordinates over time + a bounce list, without re-running inference each time.

**Gotchas:** court detection may fail on your court color/angle; expect tracking dropouts — log confidence per detection.

> **Claude Code prompt:** "Wrap the TennisProject inference into src/perception/ modules with a clean API: court_detection returns keypoints + homography, ball_tracking returns per-frame ball pixel coords, bounce_detection returns bounce frame indices. Add a homography.py helper to map pixel→court-meters. Cache all outputs to data/ as JSON so downstream stages don't re-run the models. Write pytest tests for homography.py using synthetic point correspondences."

---

### Phase 2 — In/Out engine (custom)
**Goal:** given a bounce point, decide in/out per tennis line rules.

**Tasks**
- Model the reference court in meters: baselines, singles/doubles sidelines, service boxes, center line.
- Map each bounce pixel → court coordinates via the homography.
- Line logic with a tolerance band; select which lines apply from shot context (serve → service box; rally → singles or doubles court).
- Emit a **confidence** per call from distance-to-line + tracking quality near the bounce.

**Done when:** unit-tested in/out calls pass on a set of labeled bounce points, including near-line cases.

**Gotchas:** applicable lines depend on serve-vs-rally and singles-vs-doubles; homography error grows on the far side of the court — weight confidence accordingly.

> **Claude Code prompt:** "Implement src/analysis/in_out.py. Define the tennis court reference model in meters in src/scoring/rules.py. Given a bounce point in court coordinates plus shot context (serve/rally, singles/doubles), classify in/out with a tolerance band and a confidence score. Write thorough pytest cases covering service-box bounces, baseline/sideline near-line calls, and clearly-out points."

---

### Phase 3 — Speed
**Goal:** per-shot ball speed estimates in km/h.

**Tasks**
- Displacement between consecutive court-coordinate positions ÷ frame dt.
- Smooth the trajectory; compute shot speed on the post-contact segment (or peak).
- Sanity-check against known serve speeds; surface an uncertainty range, not a false-precision number.

**Done when:** speeds are physically plausible on test clips and clearly labelled as estimates.

**Gotchas:** monocular 2D projection *underestimates* true 3D speed; low fps undersamples fast serves; smoothing choice materially changes the number.

> **Claude Code prompt:** "Implement src/analysis/speed.py: from cached ball court-coordinates and video fps, estimate per-shot speed in km/h with trajectory smoothing, returning a value plus an uncertainty band. Add a calibration note in docs/decisions.md about monocular/fps limits. Unit-test with synthetic constant-velocity trajectories."

---

### Phase 4 — Scoring engine (custom — your differentiator)
**Goal:** rally segmentation + a tennis game/set/match state machine with hybrid confirmation.

**Tasks**
- Rally segmentation from ball motion + bounces (start = serve; end = out / net / double bounce / no return).
- Scoring state machine: 0/15/30/40/deuce/advantage, games, sets, tiebreak, serve rotation.
- Wire rally outcomes (from Phase 2) into score transitions.
- Hybrid hook: propose the outcome, let the user correct; the correction drives the state update.

**Done when:** the state machine passes unit tests for full-match scenarios (incl. deuce and tiebreak) and runs over a multi-rally clip with user confirmation.

**Gotchas:** reliably detecting *why* a rally ended is the hard part — lean on the human-in-loop confirmation rather than forcing full automation.

> **Claude Code prompt:** "Implement src/scoring/state_machine.py as a pure, fully-tested tennis scoring engine (points, deuce/advantage, games, sets, tiebreak, serve rotation) with no ML dependency. Then src/analysis/rally_segmentation.py to split cached tracking data into rallies. Wire rally outcomes into the state machine with a confirm/override step. Write pytest cases for deuce, tiebreak, and a full simulated set."

---

### Phase 5 — Application layer
**Goal:** a usable app — upload video, process, view results.

**Tasks**
- `src/pipeline.py` orchestrates all stages with caching.
- FastAPI backend: upload endpoint + async processing job with progress.
- React frontend: upload, progress, results — annotated video, mini-court overlay, in/out calls, speeds, live score.
- Mini-court + stats panel (borrow the abdullahtarek visualization approach).

**Done when:** end-to-end from browser upload to results on a test clip.

**Gotchas:** processing is slow → async jobs + progress UI; handle large video uploads.

> **Claude Code prompt:** "Build src/pipeline.py to run all stages with caching and a single entrypoint. Then a FastAPI backend in app/backend with an upload endpoint and an async job that reports progress, and a React frontend in app/frontend to upload a clip, show progress, and render results: annotated video, mini-court overlay, in/out calls, per-shot speeds, and running score."

---

### Phase 6 — Polish & portfolio
**Goal:** robust, documented, demoable.

**Tasks:** error handling and edge cases; README with GIFs; a short demo video; a performance pass (lighter backbone if inference is slow); deploy a demo.

**Done when:** someone else can clone and run it, and it's portfolio-ready.

---

## 7. Milestones & sequencing

1. **M0 — "It runs":** Phase 0 done, pretrained pipeline produces output on a sample.
2. **M1 — "It sees my court":** Phase 1, clean court-coordinate trajectory + bounces on your footage.
3. **M2 — "It calls the lines":** Phases 2–3, tested in/out + speed.
4. **M3 — "It keeps score":** Phase 4, hybrid scoring over a real clip.
5. **M4 — "It's an app":** Phase 5, browser end-to-end.
6. **M5 — "It's shippable":** Phase 6.

M0→M1 is where most surprises live (footage quality, court detection on your surface). Budget generously there; the later phases are more predictable because they're mostly your own logic.

---

## 8. Key risks & mitigations

- **Monocular depth limit** — single camera sees 2D; true 3D speed and net-clearance are out of reach. *Mitigation:* scope speed as an explicit estimate with an uncertainty band; don't promise 3D.
- **Frame rate / motion blur** — a 30 fps clip undersamples a fast serve; quality caps everything. *Mitigation:* standardize on ≥60 fps footage from a fixed tripod before tuning anything.
- **Court detection on your surface** — models trained on broadcast footage may struggle on your court color/angle. *Mitigation:* validate early on real footage (M1); be ready to fine-tune the court detector.
- **Scoring reliability** — auto-detecting *why* a rally ended is research-grade. *Mitigation:* hybrid human-in-loop from day one; treat full automation as a stretch goal.
- **Scope creep** — SwingVision is years of tuning + dedicated hardware. *Mitigation:* hold the line on §1 non-goals until v1 ships.

---

## 9. First session — exact starting steps

1. Create the repo and `CLAUDE.md` (§5) with scope, architecture, and a "current phase: Phase 0" line.
2. Drop this file in as `docs/development-plan.md`.
3. Run the Phase 0 Claude Code prompt to scaffold and produce `SETUP.md`.
4. Grab **one short rally clip** at 1280×720 for `data/samples/`.
5. Get the pretrained pipeline producing output on it. Don't write a line of custom code until M0 is green.
