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
-   `config.overlay.yaml`

The system automatically loads and merges all `*.yaml` files from the `config/` directory at startup.

---

## 🏃‍♀️ Running the Orchestration

To run the entire analysis from start to finish, the command remains simple:

```bash
python3 run_pipeline.py
```

The script will automatically find and load all configurations from the `config/` directory.

To run a specific part of the orchestration, simply edit the `steps` section in `config/main.yaml`. For example, to only re-run the final video overlay without re-running detection or tracking, you would set:

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

The system is designed to be idempotent. Outputs from each step are saved to the `output_dir`, and subsequent steps will automatically use these files as inputs. This makes it efficient to re-run parts of the orchestration without reprocessing everything.
