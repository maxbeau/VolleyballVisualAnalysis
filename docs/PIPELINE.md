# 🏐 Orchestration Architecture Guide

This document provides a detailed overview of the refactored `VolleyballVisualAnalysis` orchestration system, focusing on the new configuration structure and the unified `run_pipeline.py` executor.

---

## 🚀 Core Concepts

The new architecture is built on two key principles:

1.  **Modular Configuration**: All configuration is split into logical YAML files within the `config/` directory. This makes settings easier to find, manage, and version control.
2.  **Unified Executor**: A single script, `run_pipeline.py`, remains the only entry point required to run any or all parts of the analysis. It now delegates the heavy lifting to a dedicated `Orchestrator` class.

---

## ⚙️ Configuration Structure

The `config/` directory is the heart of the new system. It contains a collection of YAML files that define the settings for each module.

### `config/main.yaml`

This file contains the global settings and the master switches for each stage.

```yaml
# config/main.yaml
global:
  video_path: "input/sample.mp4"
  output_dir: "outputs"
  cache_dir: ".cache"
  min_confidence: 0.2

steps:
  detection: true
  court_processing: true
  # ... etc.
```

### Module Configurations

Each major step has its own configuration file, such as:

-   `config/detection.yaml`
-   `config/court.yaml`
-   `config/players.yaml`
-   `config/trajectory_analysis.yaml`
-   `config/overlay.yaml`

Key highlights for the updated court tracker configuration include a new `fallback` block that enables an ECC-based motion refinement when optical flow + RANSAC struggle, as well as automatically exposing the net vanishing-point estimate used for stabilising net height. 另外，Kalman 段现在直接对 8 个同伦参数执行滤波，从而在几何上约束四角点并抑制慢漂移。

The system automatically loads and merges all `*.yaml` files from the `config/` directory at startup.

`config/detection.yaml` also supports `detection.targets`, which limits explicit cloud/local inference to the targets needed for the current run. The default analysis target set is `court`, `players`, and `ball`.

---

## 🏃‍♀️ Running the Orchestration

To run the entire analysis from start to finish, the command remains simple:

```bash
python3 run_pipeline.py
```

The script will automatically find and load all configurations from the `config/` directory.

To run a specific part of the orchestration, either edit the `steps` section in `config/main.yaml` or use the CLI override. For example, to only request the final video overlay:

```yaml
# config/main.yaml
steps:
  detection: false
  court_processing: false
  court_homography: false
  players_tracking: false
  trajectory_analysis: false
  overlay: true
```

Then, run the same command:

```bash
python3 run_pipeline.py
```

Or use:

```bash
python3 run_pipeline.py --steps overlay --cache-policy cache_only
```

The orchestrator still resolves the dependencies needed by the requested step. When `global.reuse_artifacts` is true, dependency steps reuse existing outputs under `outputs/<video_name>_<hash>/` instead of recomputing them. This is the intended loop for cloud inference: run detection once, then iterate locally on trajectory and overlay.

Detection frame caches are keyed by backend, model ID, confidence, sampling FPS, and frame cap, so changing inference settings creates a separate cache signature instead of silently mixing old model results.
