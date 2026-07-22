# Decisions log

Running log of choices made and why, so they survive across Claude Code sessions.

## 2026-07-22 — Compute: separate NVIDIA machine, not local
This dev machine has an AMD Radeon RX 580 (no CUDA support; ROCm doesn't cover this
card on Windows). All GPU inference (Phase 0 pipeline validation, Phase 1 perception
extraction) runs on a separate NVIDIA machine. Claude Code writes and tests code in
this environment; the user runs inference elsewhere and pastes back results/errors.
See `CLAUDE.md` → Compute.

## 2026-07-22 — Don't reuse TennisProject's pinned dependency versions as-is
`yastrebksv/TennisProject`'s own `requirements.txt` pins `torch==1.5.0` /
`torchvision==0.6.0` (2020-era) plus old numpy/opencv/scipy versions. These predate
modern CUDA/driver support and won't install cleanly against a current NVIDIA driver.
`requirements.txt` in this repo instead specifies `torch>=2.1`/`torchvision>=0.16`
(install via the CUDA-matched command from pytorch.org) and loosely-pinned versions
for the rest. If Phase 0 inference produces incorrect (not crashing) output, revisit
this — the original repo may implicitly depend on old-numpy/old-opencv behavior.
