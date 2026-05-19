# Volleyball Visual Analysis

Professional volleyball video analysis pipeline for court localization, player tracking, ball trajectory reconstruction, and tactical overlay rendering.

> **Powered by [Ultralytics YOLO](https://www.ultralytics.com/)**  
> This project publicly thanks the Ultralytics team for making YOLO practical, fast, and accessible for production-grade sports vision workflows.  
> Website: [ultralytics.com](https://www.ultralytics.com/) | GitHub: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

## Highlights

| Capability | Current implementation |
| --- | --- |
| Court understanding | Ultralytics Platform endpoint for court detection, followed by tracking, homography, and net estimation |
| Player analysis | Ultralytics Platform endpoint for player detection, followed by stabilized ID tracking |
| Ball path reconstruction | Roboflow ball detections plus local trajectory filtering, pseudo-3D projection, and segmentation |
| Review output | 1920x1080 overlay video, bird's-eye trajectory view, JSONL/CSV artifacts, and timeline diagnostics |
| Iteration model | Cloud inference once, then repeated local tuning through cached artifacts |

## Demo

![Full overlay preview](docs/assets/full_overlay_preview.gif)

The preview above is generated from `outputs/Volleyball_Lube_-_Verona_9751b65f/full_overlay.mp4` and shows the current end-to-end overlay: court geometry, player tracks, ball trajectory, event segments, and review HUD in one synchronized view.

Generated review artifacts include:

| Artifact | Purpose |
| --- | --- |
| `full_overlay.mp4` | Final 1920x1080 review overlay |
| `court_tracking.jsonl` | Frame-level court corners and net state |
| `players_tracks.jsonl` | Stabilized player tracks with stale prediction filtering |
| `ball_trajectory.jsonl` / `ball_trajectory.csv` | Reconstructed ball path for downstream analysis |
| `ball_path_birdseye.jpg` | Bird's-eye trajectory visualization |
| `ball_segments_timeline.png` | Trajectory segmentation and event review |

## Inference Strategy and Speed

The project optimizes for useful review speed rather than brute-force full-frame inference. Expensive visual detections are sampled per target, cached by model signature, and reused by local tracking and visualization stages.

| Target | Backend | Current model | Sampling cadence | Why this is fast enough |
| --- | --- | --- | ---: | --- |
| Court | Ultralytics Platform | `volleyball-court-1` | 3 FPS | Court geometry is stable; tracking and homography fill frame-to-frame motion locally |
| Players | Ultralytics Platform | `volleyball-players-1` | 6 FPS | Player detections anchor local tracking without paying for every video frame |
| Ball | Roboflow | `volleyball_v2/3` | 12 FPS | Ball motion changes quickly, so it receives the densest detection cadence |
| Actions | Roboflow, optional | `volleyball-actions/4` | 2 FPS | Action labels are review aids and do not block the core overlay |
| Local YOLO | Ultralytics package, optional | configured weights | user-defined | Useful for private/offline runs when suitable weights are available |

The default workflow is therefore:

1. Run cloud detections only for the targets needed by the analysis.
2. Store detections under `.cache/videos/<video_hash>/detection/<target>/<signature>/`.
3. Re-run court tracking, player tracking, trajectory analysis, and overlay rendering locally without duplicate inference spend.

This is the practical speed comparison that matters for the project today:

| Workflow | Inference cost profile | Iteration speed profile | Best use case |
| --- | --- | --- | --- |
| Ultralytics Platform for court/players | Paid once per uncached sampled frame | Fast local re-renders after cache warm-up | Production-like review loops and shared cloud endpoints |
| Local Ultralytics YOLO | No cloud call after weights are available | Hardware-dependent; excellent on GPU/MPS, slower on CPU | Private clips, offline experiments, model-debug loops |
| PaddlePaddle/PaddleDetection | Requires separate model export and runtime integration | Additional adapter and deployment work before comparable iteration | Paddle-native organizations or China-localized deployment stacks |
| MMDetection | Powerful but heavier configuration/runtime surface | Slower project onboarding for this lightweight pipeline | Research-heavy detector experimentation |

> Note: the table describes the current project workflow and integration cost, not a universal benchmark across all models, devices, and datasets. Reproducible latency benchmarks should be added once a fixed evaluation clip set and hardware profile are defined.

## Why Ultralytics

Ultralytics is the preferred path for the high-impact spatial detections in this project because it gives the best balance of engineering speed, production ergonomics, and model quality for a small team.

| Decision factor | Ultralytics YOLO | PaddlePaddle / PaddleDetection | MMDetection |
| --- | --- | --- | --- |
| Time to usable detector | Very short: train, deploy, call, normalize | Moderate: strong ecosystem, but more project-specific integration | Moderate to high: excellent research toolkit, heavier config surface |
| Python integration | Simple `ultralytics` package for local YOLO and clean HTTP Platform path | Requires Paddle runtime choices and deployment conventions | Requires MMEngine/MMCV stack alignment |
| Deployment ergonomics | Cloud endpoint and local weights are both supported in this repository | Strong if the organization is already Paddle-native | Strong for research and custom detector stacks |
| Commercial practicality | Lower integration overhead, easier hiring/onboarding, faster demo-to-product loop | Good for Paddle-centered environments; less direct for this codebase | Excellent flexibility, but more maintenance for this use case |
| Fit for this project | Best fit for court/player detection and rapid iteration | Not selected because the current need is lean integration, not ecosystem migration | Not selected because configurability would add complexity before it pays off |

The selection follows three engineering principles:

- **KISS:** use one simple YOLO path for both hosted and local detector workflows.
- **YAGNI:** avoid integrating heavyweight training frameworks until the project has a measured need for them.
- **DRY:** normalize provider outputs into the existing `predictions` schema so downstream tracking and overlay code stay provider-agnostic.

## Architecture

```text
.
├── run_pipeline.py          # CLI entrypoint
├── config/                  # Split YAML configuration
│   ├── main.yaml            # video path, enabled steps, artifact reuse
│   ├── detection.yaml       # backend, model IDs, targets, sampling FPS
│   ├── court.yaml
│   ├── players.yaml
│   ├── trajectory_analysis.yaml
│   └── overlay.yaml
├── detection/               # Roboflow, Ultralytics Platform, and local YOLO backends
├── court/                   # court tracking, homography, net support
├── players/                 # player tracking
├── ball/                    # ball selection, pseudo-3D trajectory, segmentation
├── visualization/           # final video overlay
└── docs/PIPELINE.md         # architecture notes
```

## Installation

Use Python 3.10-3.12. The Roboflow `inference-sdk` dependency does not currently support Python 3.13+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Cloud API keys are loaded from environment variables. For local development, copy `.env.example` to `.env` and fill in only the keys you need:

```bash
cp .env.example .env
```

The same values can also be exported directly in your shell:

```bash
export ROBOFLOW_API_KEY="..."
export ULTRALYTICS_API_KEY="..."
```

Never commit `.env`; it is already ignored by `.gitignore`.

For local YOLO inference, install the optional dependency and configure weights in `config/detection.yaml`:

```bash
pip install -e ".[local-yolo]"
```

## Quick Start

Edit `config/main.yaml`:

```yaml
global:
  video_path: "input/input.mp4"
  output_dir: "outputs"
  cache_dir: ".cache"
  reuse_artifacts: true
```

Run the full player-and-ball analysis:

```bash
python3 run_pipeline.py
```

Preview the resolved workflow without running inference:

```bash
python3 run_pipeline.py --dry-run
```

Override common experiment settings from the CLI:

```bash
python3 run_pipeline.py --video input/game.mp4 --targets court,players,ball
python3 run_pipeline.py --steps detection,trajectory_analysis,overlay
python3 run_pipeline.py --steps overlay --cache-policy cache_only
```

## Cloud Inference Workflow

The default cloud workflow is intentionally narrow and currently uses a hybrid provider setup:

| Target | Provider | Model |
| --- | --- | --- |
| `court` | Ultralytics Platform | `volleyball-court-1` |
| `players` | Ultralytics Platform | `volleyball-players-1` |
| `ball` | Roboflow | `volleyball_v2/3` |
| `actions` | Roboflow, optional | `volleyball-actions/4` |
| `net` | Local estimate | Estimated from court-related data where available |

`config/detection.yaml` defaults to `targets: ["court", "players", "ball"]`. `actions` and `net` detection are not run unless explicitly enabled.

The Ultralytics court/player override is configured like this:

```yaml
detection:
  backend: "roboflow"
  target_backends:
    court: "ultralytics"
    players: "ultralytics"
  models_ultralytics:
    court: "volleyball-court-1"
    players: "volleyball-players-1"
```

Set `ULTRALYTICS_API_KEY` in `.env`; do not commit credentials.

A practical iteration loop:

```bash
# 1. Pay for cloud detections once.
python3 run_pipeline.py --steps detection --targets court,players,ball

# 2. Iterate locally on trajectory and overlay.
python3 run_pipeline.py --steps trajectory_analysis,overlay --cache-policy cache_only

# 3. Re-render overlay only after visual changes.
python3 run_pipeline.py --steps overlay --cache-policy cache_only
```

To reproduce the current demo-style full run on a local clip:

```bash
python3 run_pipeline.py \
  --video "input/game.mp4" \
  --steps detection,trajectory_analysis,overlay \
  --targets court,players,ball,actions
```

For faster iteration after detections are cached:

```bash
python3 run_pipeline.py \
  --video "input/game.mp4" \
  --steps court_processing,court_homography,players_tracking,trajectory_analysis,overlay \
  --cache-policy cache_first
```

## Configuration

Use `config/main.yaml` for run-level switches:

| Key | Description |
| --- | --- |
| `global.video_path` | Input match video |
| `global.output_dir` | Per-video output root |
| `global.cache_dir` | Detection cache root |
| `global.reuse_artifacts` | Reuse existing dependency outputs when possible |
| `steps.*` | Enable or disable pipeline stages |

Use `config/detection.yaml` for inference behavior:

| Key | Description |
| --- | --- |
| `backend` | `roboflow`, `ultralytics`, or `local-yolo` |
| `target_backends` | Optional per-target backend overrides |
| `targets` | Detection targets to run when `steps.detection` is enabled |
| `cache_policy` | `cache_first`, `cache_only`, or `always_infer` |
| `infer_fps` | Target-specific sampling rates |
| `models_roboflow` / `models_ultralytics` / `models_yolo` | Model IDs or local weight paths |

## Engineering Notes

- Added a requests-based Ultralytics backend that normalizes Platform responses into the existing `predictions` schema.
- Added per-target backend selection so each detection target can use the strongest available provider.
- Hardened Ultralytics networking with endpoint-specific `NO_PROXY` handling, mirroring the existing Roboflow proxy workaround.
- Fixed stale player-track rendering by separating internal track retention from downstream output validity.
- Fixed ball trajectory export so optional height-debug metadata is not required.
- Added `.env.example` for unified cloud-key setup.

## Recommended Next Iteration

Add a small evaluation set before heavy parameter tuning:

- 3-5 short clips that represent common camera angles and rallies;
- a lightweight annotation file for ball positions, player boxes, and touch events;
- a script that writes `metrics.json` after each run;
- a reproducible speed benchmark that records hardware, backend, sampled frames, wall time, and cache policy.

That closes the loop from "the overlay looks better" to measurable detection, analysis, and performance quality.
