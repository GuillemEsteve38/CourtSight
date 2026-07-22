# Phase 0 Setup — Run the pretrained TennisProject pipeline

This machine (AMD RX 580, no CUDA) cannot run inference. All steps below marked
**[NVIDIA machine]** must be run on a separate machine with an NVIDIA GPU. Steps
marked **[either machine]** are just file management and can be done anywhere.

## 1. Clone the reference repo **[NVIDIA machine]**

```bash
git clone https://github.com/yastrebksv/TennisProject.git vendor/TennisProject
cd vendor/TennisProject
```

We treat this as a vendored reference, not a dependency of this repo — we read its
code to build our own `src/perception/` modules in Phase 1, but Phase 0 just proves
the pipeline runs.

## 2. Install PyTorch matched to your CUDA version **[NVIDIA machine]**

Check your driver/CUDA version first:

```bash
nvidia-smi
```

Then get the matching install command from https://pytorch.org/get-started/locally/
(pick Stable, your OS, Pip, Python, and the CUDA version `nvidia-smi` reports). Example
for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Do **not** use the `torch==1.5.0` pin from TennisProject's own `requirements.txt` —
it predates modern CUDA/driver support.

## 3. Install remaining dependencies **[NVIDIA machine]**

From this repo's root (copy or clone this repo onto the NVIDIA machine too):

```bash
pip install -r requirements.txt
```

## 4. Download pretrained weights **[NVIDIA machine]**

All three are Google Drive links — download manually in a browser (gdown often gets
rate-limited on these) and place them under `data/weights/`:

| Model | Link | Suggested filename |
|---|---|---|
| Ball tracking (TrackNet) | https://github.com/yastrebksv/TrackNet — see repo README for current weight link | `data/weights/tracknet.pth` |
| Court detection | https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view | `data/weights/court_detector.pth` |
| Bounce detection (CatBoost) | https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view | `data/weights/bounce_detector.cbm` |

**Gotcha:** the ball-tracking weight link lives in the TrackNet repo, not TennisProject
directly — check that repo's README for the current Drive link before downloading, in
case it's moved.

## 5. Get a sample clip **[either machine]**

Drop one short rally clip at **1280×720** into `data/samples/` (this is the resolution
TennisProject's models expect — resize/letterbox if your source differs). Per the
development plan, don't worry about your own footage yet; a downloaded/trimmed
broadcast clip is fine for Phase 0.

## 6. Run inference **[NVIDIA machine]**

```bash
python vendor/TennisProject/main.py \
  --path_ball_track_model data/weights/tracknet.pth \
  --path_court_model data/weights/court_detector.pth \
  --path_bounce_model data/weights/bounce_detector.cbm \
  --path_input_video data/samples/<your_clip>.mp4 \
  --path_output_video data/samples/<your_clip>_annotated.mp4
```

## 7. Verify **[either machine]**

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
