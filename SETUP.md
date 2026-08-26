# Phase 0 Setup — Run the pretrained TennisProject pipeline

This machine has a local NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM, driver reports
CUDA 13.1), so everything below runs here — no separate inference machine needed.

## 1. Clone the reference repo

```bash
git clone https://github.com/yastrebksv/TennisProject.git vendor/TennisProject
cd vendor/TennisProject
```

We treat this as a vendored reference, not a dependency of this repo — we read its
code to build our own `src/perception/` modules in Phase 1, but Phase 0 just proves
the pipeline runs.

## 2. Create and use a virtual environment

This project needs a Python 3.11 runtime — a newer interpreter on this machine (3.14)
breaks the scientific stack the pipeline depends on.

Install Python 3.11 first (for example from python.org or the Microsoft Store), then create the environment from that interpreter:

```bash
C:\Path\To\Python311\python.exe -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

If you are using VS Code, select the interpreter at `.venv\Scripts\python.exe` so the editor uses the same environment (already configured in `.vscode/settings.json`).

## 3. Install PyTorch matched to your CUDA version

Check your driver/CUDA version first:

```bash
nvidia-smi
```

Then get the matching install command from https://pytorch.org/get-started/locally/
(pick Stable, your OS, Pip, Python, and the CUDA version `nvidia-smi` reports). This
machine's driver reports CUDA 13.1, and the cu121 wheels install and detect the GPU
fine:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Do **not** use the `torch==1.5.0` pin from TennisProject's own `requirements.txt` —
it predates modern CUDA/driver support.

## 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` already pins `torch`/`torchvision` too (unpinned version range) —
reinstalling from step 3's CUDA-specific index first, then running this, is what
gives you the CUDA build instead of pip silently grabbing a CPU-only wheel.

## 5. Download pretrained weights

All three are Google Drive links — download manually in a browser (gdown often gets
rate-limited on these) and place them under `data/weights/`:

| Model | Link | Suggested filename |
|---|---|---|
| Ball tracking (TrackNet) | https://github.com/yastrebksv/TrackNet — see repo README for current weight link | `data/weights/tracknet.pt` |
| Court detection | https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view | `data/weights/court_detector.pt` |
| Bounce detection (CatBoost) | https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view | `data/weights/bounce_detector.cbm` |

**Gotcha:** the ball-tracking weight link lives in the TrackNet repo, not TennisProject
directly — check that repo's README for the current Drive link before downloading, in
case it's moved.

## 6. Get a sample clip

Drop one short rally clip at **1280×720** into `data/samples/` (this is the resolution
TennisProject's models expect — resize/letterbox if your source differs). Per the
development plan, don't worry about your own footage yet; a downloaded/trimmed
broadcast clip is fine for Phase 0.

## 7. Run inference

```bash
.venv\Scripts\python.exe vendor/TennisProject/main.py \
  --path_ball_track_model data/weights/tracknet.pt \
  --path_court_model data/weights/court_detector.pt \
  --path_bounce_model data/weights/bounce_detector.cbm \
  --path_input_video data/samples/<your_clip>.mp4 \
  --path_output_video data/samples/<your_clip>_annotated.mp4
```

Run this from the repository root, not from inside `vendor/TennisProject/`, and make
sure the virtual environment is activated first. `main.py` auto-detects CUDA
(`device = 'cuda' if torch.cuda.is_available() else 'cpu'`) — no flag needed to use
the GPU, just make sure step 3 installed a CUDA build of torch.

## 8. Verify

Open `data/samples/<your_clip>_annotated.mp4` and confirm:
- Court lines/keypoints are overlaid correctly
- Ball trajectory is drawn and roughly tracks the visible ball
- Bounce points are marked

If all three show up, Phase 0 is done (**M0 — "It runs"**). Paste back any errors —
CUDA/driver mismatches and Google Drive download failures are the most likely
failure points (see development-plan.md §6 Phase 0 gotchas).

## Known risks going in
- `main.py`'s exact CLI args (`--path_ball_track_model`, `--path_court_model`,
  `--path_bounce_model`, `--path_input_video`, `--path_output_video`) were read from
  the repo source, not run yet — the repo may have moved on since; check
  `vendor/TennisProject/main.py --help` if this fails.
- Old pinned deps (numpy 1.19.3, opencv 4.1.2.30, etc. in TennisProject's own
  requirements.txt) may cause subtle behavior differences vs. what it was validated
  against. If inference produces garbage output (not just a crash), this is a
  suspect — note it in `docs/decisions.md` and consider pinning closer to their
  versions in a dedicated venv.
- `vendor/TennisProject/utils.py` uses the pre-1.0 PySceneDetect `VideoManager` API,
  removed in scenedetect 0.7 — `requirements.txt` pins `scenedetect==0.6.7.1` to match;
  see `docs/decisions.md`.

---

# Running the app (Phase 5)

## Backend

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --reload
```

Serves the API on http://127.0.0.1:8000. If `app/frontend/dist` exists it also serves the
built UI at `/`, so this single command is enough to use the app.

Calling the venv's `python.exe` directly avoids needing to activate the venv first — a bare
`uvicorn ...` only works in an activated shell, and otherwise fails with
`'uvicorn' is not recognized`. `python -m` also puts the repo root on `sys.path`, which is
what lets `app.backend` and `src.*` import; running uvicorn from any other directory will
fail with `ModuleNotFoundError`.

`GET /api/health` reports whether CUDA was found — check it first if analysis is slow.

## Frontend

The build output is committed-ignored, so build it once:

```bash
cd app/frontend
npm install
npm run build
```

Then reload the backend and open http://127.0.0.1:8000.

For frontend development with hot reload, run the backend as above and in another terminal:

```bash
cd app/frontend
npm run dev
```

That serves the UI on http://127.0.0.1:5173 and proxies `/api` to the backend.

## Command line

The app is not required — the same pipeline runs headlessly:

```bash
# extract + cache perception output (court keypoints, ball track, bounces)
.venv\Scripts\python.exe scripts/run_perception.py --input data/samples/<clip>.mp4

# render an annotated debug video from that cache
.venv\Scripts\python.exe scripts/render_debug_overlay.py --input data/samples/<clip>.mp4

# score a clip interactively, confirming each proposed rally outcome
.venv\Scripts\python.exe scripts/score_match.py --input data/samples/<clip>.mp4
```

Input at any 16:9-ish resolution is resized to 1280×720 automatically. Anything further
from 16:9 is rejected with a clear message rather than silently distorted — see
`docs/decisions.md`.
