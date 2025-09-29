# 🏐 Pipeline Architecture Guide

This document provides a detailed overview of the refactored `VolleyballVisualAnalysis` pipeline, focusing on the `pipeline.yaml` configuration file and the unified `run_pipeline.py` executor.

---

## 🚀 Core Concepts

The new architecture is built on two key principles:

1.  **Single Source of Truth**: All configuration for every stage of the pipeline is consolidated into a single file: `pipeline.yaml`. This eliminates the need for scattered `.env` files and command-line arguments.
2.  **Unified Executor**: A single script, `run_pipeline.py`, is the only entry point required to run any or all parts of the analysis. It reads `pipeline.yaml` and orchestrates the execution of the configured steps in the correct order.

---

## ⚙️ `pipeline.yaml` Structure

The `pipeline.yaml` file is the heart of the new system. It is a YAML file that defines the settings for each module. Here is a breakdown of its top-level sections:

### `global`

This section contains settings that apply to the entire pipeline.

```yaml
global:
  video_path: "input/sample.mp4"
  output_dir: "outputs"
  cache_dir: ".cache"
  min_confidence: 0.2
  show_box_labels: true
```

-   `video_path`: **(Required)** Path to the input video file.
-   `output_dir`: Directory where all generated files (detections, tracks, videos) will be saved.
-   `cache_dir`: Root directory for intermediate caches. Each run now uses
    `.cache/videos/<video_stem>_<hash>/...`, keeping different inputs separated
    automatically.
-   `min_confidence`: A global confidence threshold for detections.
-   `show_box_labels`: A global switch to show or hide labels on bounding boxes in the final overlay.

### `steps`

This section acts as a set of on/off switches for each stage of the pipeline. Set a step to `true` to run it, or `false` to skip it.

```yaml
steps:
  detection: true
  court_processing: true
  court_homography: true
  players_tracking: true
  trajectory_analysis: true
  overlay: true
```

### Module Configurations

Each of the following sections corresponds to a specific step in the pipeline and configures its behavior.

-   **`detection`**: Configures the object detection backend (Roboflow or local YOLO).
-   **`court`**: Settings for court line tracking and homography calculation.
-   **`players`**: Parameters for the player tracking algorithm (ByteTrack).
-   **`trajectory_analysis`**: Configuration for ball trajectory smoothing and world coordinate mapping.
-   **`overlay`**: Detailed settings for customizing the final output video, including colors, tail effects, and HUD elements.

For a complete, annotated example of `pipeline.yaml`, please refer to the root of the repository.

---

## 🏃‍♀️ Running the Pipeline

To run the entire analysis from start to finish, you only need a single command:

```bash
python3 run_pipeline.py
```

The script will automatically find and load `pipeline.yaml` from the project root.

To run a specific part of the pipeline, simply edit the `steps` section in `pipeline.yaml`. For example, to only re-run the final video overlay without re-running detection or tracking, you would set:

```yaml
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

The pipeline is designed to be idempotent. Outputs from each step are saved to the `output_dir`, and subsequent steps will automatically use these files as inputs. This makes it efficient to re-run parts of the pipeline without reprocessing everything.
