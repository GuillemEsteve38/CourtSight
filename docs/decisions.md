# Decisions log

Running log of choices made and why, so they survive across Claude Code sessions.

## 2026-07-22 — Compute: separate NVIDIA machine, not local
This dev machine has an AMD Radeon RX 580 (no CUDA support; ROCm doesn't cover this
card on Windows). All GPU inference (Phase 0 pipeline validation, Phase 1 perception
extraction) runs on a separate NVIDIA machine. Claude Code writes and tests code in
this environment; the user runs inference elsewhere and pastes back results/errors.
See `CLAUDE.md` → Compute.

**Superseded 2026-07-31** — this assumption was wrong for the actual dev machine.
`Get-CimInstance Win32_VideoController` and `nvidia-smi` both show a local NVIDIA
GeForce RTX 3070 Laptop GPU (8GB VRAM, driver CUDA 13.1) alongside AMD integrated
graphics — the RX 580 described here isn't present. Installed CUDA-enabled torch
(`torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`) directly in `.venv`; `torch.cuda.is_available()`
returns `True` and resolves to the RTX 3070. No separate NVIDIA machine is needed —
`vendor/TennisProject/main.py` already auto-selects `cuda` when available
(`device = 'cuda' if torch.cuda.is_available() else 'cpu'`), so no code changes were
needed, just the dependency swap. `CLAUDE.md` → Compute and `SETUP.md` updated to
match. Unclear whether the RX 580 assumption came from a different machine or was
simply never verified — worth double-checking before trusting other unverified specs
in the docs.

## 2026-07-22 — Don't reuse TennisProject's pinned dependency versions as-is
`yastrebksv/TennisProject`'s own `requirements.txt` pins `torch==1.5.0` /
`torchvision==0.6.0` (2020-era) plus old numpy/opencv/scipy versions. These predate
modern CUDA/driver support and won't install cleanly against a current NVIDIA driver.
`requirements.txt` in this repo instead specifies `torch>=2.1`/`torchvision>=0.16`
(install via the CUDA-matched command from pytorch.org) and loosely-pinned versions
for the rest. If Phase 0 inference produces incorrect (not crashing) output, revisit
this — the original repo may implicitly depend on old-numpy/old-opencv behavior.

## 2026-07-31 — Pin scenedetect to 0.6.7.1
`vendor/TennisProject/utils.py`'s `scene_detect()` imports `scenedetect.video_manager.VideoManager`,
`SceneManager`, `StatsManager`, and `ContentDetector` from the old (pre-1.0) PySceneDetect API.
PySceneDetect 0.7 removed `VideoManager` entirely as part of a larger API rewrite (`open_video()` +
`SceneManager` is the new pattern), so an unpinned `scenedetect` install (latest was 0.7.1) fails with
`ModuleNotFoundError: No module named 'scenedetect.video_manager'`. Pinned to `scenedetect==0.6.7.1`
(last 0.6.x release) in `requirements.txt`, which still has `VideoManager` and imports cleanly. Left
`vendor/TennisProject/utils.py` unmodified since it's a vendored reference, not our code — if Phase 1
reimplements scene detection in `src/perception/`, use the modern `open_video()`/`SceneManager` API
instead of carrying this pin forward.

## 2026-07-31 — Also pin opencv-python to match opencv-python-headless
`requirements.txt` pinned `opencv-python-headless<4.11`, but `scenedetect` depends on plain
`opencv-python` (not headless), so pip resolved an unpinned `opencv-python==4.11.0.86` alongside
it. Both packages install into the same `cv2` import namespace, so having two different versions
installed is fragile (whichever installs last wins, silently). Added `opencv-python<4.11` to
`requirements.txt` so both resolve to the same version deterministically.

## 2026-07-31 — Sample clip must be exactly 1280×720, not just any 16:9 resolution
First Phase 0 run (on `data/samples/input_video.mp4`, actually 1920×1080) produced visibly
offset court keypoints and only detected one of the two players. Root cause, confirmed by
reading `vendor/TennisProject/court_detection_net.py:infer_model()`: it resizes each frame to
640×360 for the heatmap model, then hardcodes `scale = 2` to project keypoints back up,
i.e. it assumes the original frame is exactly 1280×720. Feeding it a 1920×1080 frame (1.5×,
not 2×, larger than 640×360) put every keypoint at ~2/3 of its correct position. That bad
homography then fed `person_detector.py:detect_top_and_bottom_players()`, which classifies
each detected person as top/bottom-court by warping a court mask through the homography and
checking if the person falls inside it — with a misaligned mask, a player near the boundary
gets silently dropped instead of just misplaced. This explains both symptoms as one root
cause. Fix: resize the input to exactly 1280×720 before running inference (same aspect
ratio as the original 1920×1080 clip, so a plain `cv2.resize` works, no letterboxing needed).
Reran on the resized clip to confirm both the keypoint offset and missing player were fixed.
SETUP.md already warned "resize/letterbox if your source differs" — this is why it matters:
it's not just about matching output quality, the homography math silently breaks. Anyone
supplying their own footage in Phase 1+ must resize/letterbox to 1280×720 first; consider
adding this as an automatic preprocessing step in `src/pipeline.py` rather than relying on
users to do it manually.
