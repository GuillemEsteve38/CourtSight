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
supplying their own footage in Phase 1+ must resize/letterbox to 1280×720 first.

**Resolved 2026-07-31** — this kept recurring (hit again on `xenia.mp4` and `xenia2.mp4`,
both 848×478) since fixing it manually each time relied on remembering to. Automated it:
`src/perception/video_io.py`'s `ensure_target_resolution()` resizes in-memory frames to
1280×720 whenever the source aspect ratio is within 2% of 16:9, and raises a clear error
instead of silently distorting the court otherwise (true letterboxing for non-16:9 sources
is still unimplemented — out of scope until footage that actually needs it shows up).
`scripts/run_perception.py` calls it automatically. `vendor/TennisProject/main.py` itself is
NOT patched (still vendored, read-only reference) — anyone running it directly still needs
to pre-resize manually, same as SETUP.md describes.

## 2026-07-31 — Phase 1: own meters-based court reference instead of vendor's pixel scale
`src/perception/homography.py` defines its own `REFERENCE_KEYPOINTS_M`, a 14-point court
model in real meters (standard ITF doubles dimensions: 23.77m x 10.97m, singles inset
1.37m, service line 6.40m from net), with the same point ordering as vendor's
`CourtReference.key_points` in `vendor/TennisProject/court_reference.py`. Considered instead
reusing vendor's `CourtReference` pixel-space model (its own reference image is drawn at an
arbitrary ~101.5 px/meter scale) and converting via a fixed scale factor, but that requires
importing vendor code into what should be a pure, GPU-free, always-testable module, and
carries an extra indirection (video-px → vendor-ref-px → meters) for no benefit — verified
by reverse-engineering vendor's own pixel constants that they already correspond almost
exactly to real ITF measurements (e.g. their service-line-to-net distance converts to
6.39m vs the regulation 6.40m), so defining our own meters model directly is equivalent and
simpler. Consequence: `src/perception/court_detection.py` does NOT reuse vendor's own
`homography_matrices` output (video-px → vendor-ref-px, inverted internally in
`court_detection_net.py`) at all — it only keeps vendor's reprojected `kps_court` (14
keypoints reprojected into video-pixel space) and computes its own pixel→meters homography
via `estimate_homography()`. If Phase 2's in/out geometry ever needs vendor's exact internal
scale for cross-checking, it's recoverable from `court_reference.py`'s constants, but there's
no known reason to.

## 2026-07-31 — Phase 1: vendor code imported via sys.path, not as an installed package
`vendor/TennisProject`'s files use flat imports (e.g. `from tracknet import BallTrackerNet`)
that only resolve if the vendor directory itself is on `sys.path` — they're a vendored
reference script collection (see CLAUDE.md), not a pip-installable package, and per
CLAUDE.md/SETUP.md conventions we don't modify vendored code. `src/perception/_vendor_bridge.py`
inserts `vendor/TennisProject` onto `sys.path` once, called by `court_detection.py`,
`ball_tracking.py`, and `bounce_detection.py` before their vendor imports. Tradeoff: this
mutates global interpreter state (`sys.path`) and vendor module names (`tracknet`,
`homography`, `utils`, etc.) become globally importable bare names for the rest of the
process, which could theoretically shadow an unrelated same-named module elsewhere in this
project — acceptable here since this is a small, self-contained repo and the alternative
(vendoring TennisProject as a proper installable package, or copying/adapting its source
into `src/`) is much more invasive for a pipeline we're already told to treat as read-only
upstream reference. `src/perception/homography.py` itself has no vendor dependency at all
(see the decision above), so the one module that must stay GPU/weights-free for testing is
unaffected by this.

## 2026-07-31 — Phase 2: in/out confidence model and service-box scope
`src/analysis/in_out.py`'s `classify_bounce()` computes a signed distance from the bounce
point to the relevant court region (`src/scoring/rules.py`'s axis-aligned `Rect`s) and maps
it to a confidence in `[MIN_CONFIDENCE=0.5, 1.0]`: 0.5 right on the line (a coin flip),
ramping linearly to 1.0 at `CONFIDENCE_FULL_CERTAINTY_MARGIN_M=0.10` (10cm) away in either
direction. `tolerance_m` (default 0.05m, roughly a line's width plus homography slack) is a
separate parameter controlling where "in" vs "out" flips, so a ball just past the true
boundary but within tolerance is still called in (matching real line-touch rules) albeit
with reduced confidence. `tracking_confidence` (default 1.0) is an external multiplier the
caller supplies — the module deliberately does NOT try to estimate ball-tracking quality
itself (e.g. from gaps in the cached trajectory near the bounce frame); that's a concern of
whichever stage calls `classify_bounce()` with real data (likely `src/pipeline.py` or a
future rally-segmentation step), keeping this module pure geometry and independently
testable. Also: service boxes are modeled as always singles-width regardless of match type
(`court_type` only affects "rally" shots) -- this matches the actual ITF rule that serves
must land within the singles-width box even in doubles play. `src/scoring/rules.py`
re-defines the same four ITF constants (23.77/10.97/8.23/6.40m) already in
`src/perception/homography.py` rather than importing them, keeping perception and scoring
decoupled per the module boundaries in CLAUDE.md; both cite the same public regulation
numbers so they can't drift independently in a way that matters. Not yet handled: which
specific service box is "correct" for a given serve depends on score state (deuce/ad court)
that only Phase 4's scoring engine will know -- `classify_bounce()` takes `service_box` as
a required caller-supplied argument for serves rather than guessing it.

## 2026-07-31 — M1 achieved: pipeline validated on real user footage (not broadcast)
Ran `vendor/TennisProject/main.py` on `data/samples/xenia.mp4`, the user's own club-court
footage (two players, non-broadcast camera/lighting/angle) -- this is the actual Phase 1
goal (§6 M1 "It sees my court"), distinct from the Phase 0 sample clip which was a
downloaded ATP broadcast. Hit the exact same resolution bug as Phase 0: `xenia.mp4` is
848x478, not 1280x720, so `court_detection_net.py`'s hardcoded `scale=2`
reprojection produced out-of-bounds crop coordinates in `postprocess.py:refine_kps()` ->
`detect_lines()` -> `cv2.cvtColor` raising `(-215:Assertion failed) !_src.empty()` (crop
slice with start > stop is empty in numpy). Fix: same as before, resize to exactly
1280x720 (`xenia_720p.mp4`) -- 848x478's aspect ratio is 0.2% off from 16:9 so a plain
`cv2.resize` introduces no visible distortion. Confirms the resolution assumption is a
systemic gotcha for any footage that isn't natively 1280x720, not a one-off; worth adding
an explicit resize/letterbox check (with a clear error or auto-fix) to whatever ingests
video in `src/pipeline.py` (Phase 5), rather than relying on remembering to preprocess.

Visually verified 3 sampled frames from the annotated output: court keypoints land
precisely on real line intersections/corners despite different lighting, camera angle,
and court color/surface than the Phase 0 broadcast clip -- the risk called out in
development-plan.md §8 ("court detection may fail on your court color/angle") did not
materialize here. Both players detected and tracked throughout.

Also reran through our own `src/perception/` wrapper modules (not just vendor's
`main.py`) on the resized clip to confirm the actual code deliverable works on real
footage, not just the vendored reference script: keypoints + homography detected on
428/428 frames, ball on 287/428, 9 bounce frames found. Cached to
`data/processed/xenia_720p/`. M1 is done.

## 2026-07-31 — Phase 3: speed is measured by peak window, and refuses to answer when it can't
`src/analysis/speed.py` reports km/h **with an uncertainty band**, never a bare number.
Two error sources are modelled explicitly:

* Random (`DEFAULT_POSITION_UNCERTAINTY_M = 0.15`): at 30 fps one frame pair spans 33 ms,
  so 15 cm of position error alone is ~20 km/h. Speed is therefore measured
  endpoint-to-endpoint across a window (error ~ sigma*sqrt(2)/T, shrinking with window
  duration) rather than by differencing consecutive frames, which amplifies noise at
  every step.
* Systematic (`MONOCULAR_UNDERESTIMATE_ALLOWANCE = 0.15`): a single camera recovers only
  ground-plane motion, so a ball with vertical travel covers more 3D distance than we
  measure. The band is asymmetric upward for this reason.

**Endpoint displacement was wrong as the per-shot number.** Between two bounces the ball
arcs out and back, so *net* displacement collapses toward zero; on real clips this
reported 0 km/h for genuine shots and 5-22 km/h for rallies. `peak_speed()` (a sliding
0.25 s window, max over the segment) is the per-shot number to report --
`peak_speeds_between_bounces()` is the intended entrypoint. `speeds_between_bounces()` is
kept for segment-average use but is not the shot speed.

**Guards added after real-data sanity checks, not from theory.** Unguarded, the module
produced `862 km/h (839-1017)` from a 2-sample, 0.03 s window: correct arithmetic on
broken tracking data, presented with a confident-looking narrow band. Added
`MAX_PLAUSIBLE_SPEED_KMH` (260; fastest recorded serve ~263), `MIN_WINDOW_S`,
`MIN_SAMPLES`, `MIN_COVERAGE` (a window whose endpoints straddle a tracking gap says
nothing about the motion between them, and `peak_speed`, being a maximum, actively seeks
such windows out), and `reject_implausible_positions()` (ball tracking sometimes locks
onto the crowd/fence, projecting 14 m behind the baseline). All are parameters, not
constants, so callers can trade safety for recall. `estimate_speed` returns None rather
than a number it cannot support -- callers must treat None as "unknown", not zero.

### Validation result: trustworthy on clean tracking, not yet on the amateur clips
Broadcast clip (`input_video_720p`, ball detected on 201/214 frames): per-flight peaks of
**138 / 137 / 143 km/h** -- textbook professional rally groundstroke speeds. Good signal
that the geometry, units, and window logic are right.

User footage (`xenia_720p` 287/428 frames, `xenia2` 338/687): peaks of 213-259 km/h, which
are not credible for club tennis and sit suspiciously just under the 260 cap. Diagnosed to
a specific window (frames 318-326): the ball moved ~13 px between frames 318 and 320 but
that mapped to **5.5 m** of court distance. Near the far baseline the homography's
meters-per-pixel is enormous, so sub-pixel jitter becomes tens of km/h; combined with
every-other-frame dropouts, that manufactures fake peaks. Tracking also jumped between the
far court and the bottom of frame, indicating lost target.

This is a property of ground-plane homography plus footage quality, **not** something more
heuristics should hide -- development-plan.md §8 already flags footage quality (>=60 fps,
fixed tripod) as the risk that caps everything downstream. Two concrete follow-ups, neither
done here:
1. **Position-dependent uncertainty.** Replace the constant 0.15 m with a per-point value
   derived from the homography Jacobian at that pixel (meters-per-pixel locally). A
   far-court measurement would then carry a correspondingly huge band and self-identify as
   low confidence, instead of reporting "248 km/h (245-289)". This is the principled fix
   and should come before any further tuning of thresholds.
2. **Better footage**: >=60 fps materially shortens the usable window and improves tracking
   density; the plan already calls for it.

Until then: treat speed as validated on dense/clean tracking only, and do not present
per-shot speeds from the current amateur clips as accurate.

## 2026-07-31 — Phase 4: scoring is a pure immutable state machine; rally outcomes are proposals only
`src/scoring/state_machine.py` is deliberately free of any ML/video dependency: every
transition is `(state, point winner) -> new state`, so the whole format -- points, deuce,
advantage, no-ad, games, sets, tiebreaks, advantage final sets, serve rotation, best-of-3/5
-- is exhaustively unit-testable (36 tests, including a full three-set match with a deuce
game and a tiebreak). This is the part of the system that must never be wrong, and it is
the part with no uncertainty in it.

**MatchState is immutable and MatchScorer keeps a history stack.** That is not stylistic:
rally outcomes come from unreliable automation, so `undo()` has to be exact, and rebuilding
a score by replaying mutations would eventually drift. Popping a stack of immutable
snapshots cannot.

**Tiebreak serve is derived, not stored.** `MatchState.server` computes the tiebreak
rotation (one point, then two each) from points played rather than tracking it as a field,
so it cannot fall out of sync with the score. Serve simply alternates on every game, which
also yields the correct post-tiebreak server (the player who served the tiebreak's first
point receives first in the next set).

### Rally outcomes: capped confidence, on purpose
`src/analysis/rally_segmentation.py` segments rallies by gaps between bounces
(>2.5s of quiet = new point) and proposes a winner from the final bounce:
a bounce in a player's half was hit by their opponent; if it landed **out** the hitter
loses; if it landed **in** we assume the receiver failed to return it.

That last inference is weak -- a rally also ends on a net cord, a miss after a good bounce,
or a bounce the detector missed -- so every proposal is multiplied by
`MAX_PROPOSAL_CONFIDENCE = 0.6`. Nothing here is ever presented as near-certain, which
matches development-plan.md §8's assessment that detecting *why* a rally ended is
research-grade. `MatchScorer` records whether a human accepted or overrode each proposal
and exposes `correction_rate` -- that number, not a confidence score we invented, is the
honest measure of how good the automation actually is.

### Bounce frames and tracked ball positions don't always coincide
Found while running the scorer end-to-end, not in tests: an entire real rally
(`xenia_720p`, frames 153-237, 4 bounces) came back "inconclusive -- no ball position at
any bounce". Cause: the CatBoost bounce detector interpolates gaps in the pixel track
internally, so it legitimately reports bounces on frames where our cached track is None.
Fixed by searching +/-3 frames for the nearest tracked position, with a confidence penalty
proportional to the offset (the ball is moving fast near a bounce, so a borrowed position
is metres from where it actually landed) and the borrow disclosed in the proposal's
`reason` string. Both clips now score end-to-end with no unscoreable rallies.

### Not done
* **Serves are scored with rally rules.** Correctly judging a serve needs the service box,
  which depends on the deuce/ad court -- i.e. on the score itself. The wiring
  (state machine -> expected service box -> `classify_bounce(shot_type="serve")`) is
  straightforward but untested against real serve footage, so it is left out rather than
  guessed at. Phase 2 already takes `service_box` as a required caller argument for exactly
  this reason.
* **Rally boundaries are unvalidated.** Both test clips segmented into 2 rallies each,
  which is plausible for their length but was not checked frame-by-frame against the video.
  The 2.5s gap threshold is a conservative guess, biased toward splitting rather than
  merging (a missing point is easier for a human to spot than a silently merged one).

## 2026-07-31 — Phase 5: pipeline split into pure/impure halves, in-process jobs, one scoring implementation
`src/pipeline.py` is deliberately two functions. `analyze_video()` does the impure work
(decode, run three models, read/write the JSON cache); `build_analysis()` turns loaded
perception data into the final result and is pure -- no torch, no disk, no GPU. That keeps
the assembly logic (which bounce gets a call, which segments produce a speed, how rallies
map to proposals) unit-testable, and lets the API rebuild a response from cache without
loading a model.

**Async jobs are in-process, not a broker.** Analysis takes tens of seconds to minutes, so
uploads return a job id and the client polls. `app/backend/jobs.py` is an in-memory dict
plus a *single-worker* thread pool: one worker because the GPU can only usefully do one
clip at a time and serialising keeps VRAM predictable, threads (not processes) because the
heavy stages are torch/OpenCV calls that release the GIL. The cost -- jobs are lost on
restart -- is the honest trade for a local tool; swapping in Celery/RQ later means
reimplementing `submit` and `get` and nothing else.

**Progress had to be driven from inside the vendored loops.** Reporting only at stage
boundaries left the bar frozen at 4% for ~70s of court detection. Neither vendor loop
accepts a callback, and they read frames differently (`court_detection_net` iterates the
list, `ball_detector` indexes it), so `_ProgressFrames` is a `list` subclass that reports
from *both* access paths -- real intra-stage progress without patching read-only vendored
code. Progress is high-watermarked because `ball_detector` reads backwards
(`frames[num-1]`, `frames[num-2]`) and a bar that goes backwards looks broken. Verified
end-to-end: progress now advances smoothly 0 -> 100 instead of in four jumps.

**Scoring lives only in Python.** The frontend could have implemented tennis scoring in JS,
but two implementations of the rules is two things to keep correct. Instead `POST /api/score`
takes the list of confirmed winners and runs the tested `MatchScorer`. It is stateless: the
client owns the list and re-posts it whenever it changes, so undo/override in the UI is just
"send a shorter list", and there is no server-side session to desynchronise.

**The upload endpoint is sync, not async.** Writing a multi-hundred-MB upload is blocking
IO; inside an `async def` it would stall the event loop and freeze the progress polling the
UI depends on -- a real bug, not a lint nit. A sync endpoint runs in FastAPI's threadpool.

### The UI states uncertainty rather than hiding it
Speeds are shown with their band (`138` / `135-162 km/h`) and labelled estimates; rally
outcomes are shown as proposals with a confidence bar that tops out around 60%, with
explicit confirm/override buttons; and a warning appears when ball tracking covers under
70% of frames, since that is the regime where Phase 3 found speeds become unreliable. This
is the same honesty the analysis modules enforce, carried through to the surface the user
actually looks at.

### Verified end-to-end
Uploaded `xenia2.mp4` through the running server: job queued, progressed through all five
stages, produced a result identical to the CLI (same 3 bounces, 2 shots, 2 rallies) and a
38 MB annotated video served over HTTP. The built frontend is served by FastAPI from `/`.

## 2026-07-31 — Phase 6: frame budget, and exploiting the fixed camera for a 2.4x speedup

### Memory is the hard limit on clip length, so it fails loudly now
`video_io` holds every decoded frame in RAM: 2.64 MB per 1280x720 frame, so ~1.8 GB for
20 seconds and ~4.6 GB per minute. On the 15.3 GB dev machine (2.6 GB free at the time of
measuring) anything over about a minute would have OOM'd or sent the machine swapping,
with no useful error. Two fixes:

* `ensure_target_resolution` now resizes **in place**. Building a second list meant the
  original and resized frames coexisted; measured peak for a 23s clip dropped from
  ~2.6 GB to 1.77 GB. It mutates its argument, which is a real side effect -- documented
  in the docstring and pinned by a test rather than left as a surprise.
* `read_frames` refuses clips above `DEFAULT_MAX_FRAMES` (2400, ~80s at 30fps, ~6.3 GB),
  checked against container metadata *before* decoding so it fails in milliseconds, and
  again while reading because metadata can lie. The error states the estimated RAM and
  how to override it.

Streaming instead of buffering is not available: the vendored models take a frame *list*,
not an iterator, so streaming means reimplementing their inference loops.

### The per-frame homography is mostly noise, and the camera never moves
Measured on `xenia2`: projecting one fixed pixel through each frame's homography moves it
with a standard deviation of **37 cm** in court metres (33.6 cm on `xenia_720p`). Checked
whether that was camera drift or detection jitter -- it is jitter: the values oscillate
around a stable mean with no trend, median frame-to-frame change 4.8 cm. Notably the
broadcast clip shows only **2.4 cm**, so this is a symptom of weak court detection on
amateur footage, not a universal property.

37 cm of noise is seven times the 5 cm tolerance the in/out engine uses to decide a line
call. Since v1 explicitly assumes a fixed camera, there is exactly *one* true homography
and the per-frame estimates are noisy samples of it. Two changes follow:

* `detect_court(sample_every=N)` runs inference on every Nth frame and reuses the nearest
  result. Court detection was by far the slowest stage: **90.4s -> 8.5s at N=10, a 10.6x
  speedup**; ball tracking is now the bottleneck, and `STAGE_WEIGHTS` was rebalanced to
  match so the progress bar stays honest.
* `consolidate_fixed_camera()` takes the per-keypoint **median** across frames (median,
  not mean, so one badly-detected frame cannot drag the court sideways) and applies the
  resulting single homography everywhere.

Both default on, and both are wrong for moving footage -- `court_sample_every=1,
fixed_camera=False` restores the old behaviour.

**Validation.** End-to-end on `xenia2`: 124.8s -> 51.5s, **2.42x faster**, with
*identical* bounce in/out calls and identical rally proposals. Across all three clips no
in/out call changed, though consolidation moved individual bounce positions by up to
63 cm on the noisy amateur clips (and by under 1.5 cm on the broadcast clip, where there
was little noise to remove). Shot speeds shifted (259 -> 226 km/h on one flight), but
both values sit inside the range already documented as unreliable for that footage, so
this is noise moving rather than a regression.

**What is not claimed:** consolidation demonstrably reduces *variance*, and averaging
noisy samples of a constant is the right thing to do for a fixed camera. But there is no
ground-truth court survey for these clips, so it is not claimed as an accuracy
improvement -- only as less noisy, with no call changed. It is plausibly relevant to the
Phase 3 speed problem, since noisy court positions were one suspected contributor, but
that remains unproven and the Jacobian-based fix is still the outstanding work.

### Portfolio work
README rewritten around real generated output (`docs/media/demo.gif` and
`annotated-frame.png` are rendered by `src/viz/`, not mocked up), and leads with what the
system cannot do plus where speeds are and are not trustworthy. A recorded walkthrough
video and any public deployment are deliberately left to the user: both involve their
footage, their accounts, and an outward-facing publishing decision.
