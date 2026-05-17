"""Command-line entrypoint for the Volleyball Visual Analysis pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from core.context import PipelineContext
from core.logger import setup_logger
from core.orchestrator import Orchestrator
from orchestration.config import load_config


STEP_NAMES = (
    "detection",
    "court_processing",
    "court_homography",
    "players_tracking",
    "trajectory_analysis",
    "overlay",
)


def _csv_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run volleyball video analysis.")
    parser.add_argument("--config-dir", default="config", help="Directory containing pipeline YAML files.")
    parser.add_argument("--video", help="Override global.video_path for this run.")
    parser.add_argument("--steps", help=f"Comma-separated steps to enable. Available: {', '.join(STEP_NAMES)}.")
    parser.add_argument("--targets", help="Comma-separated detection targets, e.g. court,players,ball.")
    parser.add_argument("--cache-policy", choices=("cache_first", "cache_only", "always_infer"))
    parser.add_argument("--no-reuse-artifacts", action="store_true", help="Force dependency steps to recompute.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved execution order without running.")
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace):
    if args.video:
        config.global_settings.video_path = Path(args.video)
    if args.no_reuse_artifacts:
        config.global_settings.reuse_artifacts = False
    if args.cache_policy and config.detection:
        config.detection.cache_policy = args.cache_policy
    if args.targets and config.detection:
        allowed_targets = {"court", "players", "ball", "actions", "net"}
        targets = _csv_values(args.targets)
        invalid = [target for target in targets if target not in allowed_targets]
        if invalid:
            raise ValueError(f"Unsupported detection targets: {', '.join(invalid)}")
        config.detection.targets = targets  # type: ignore[assignment]
    if args.steps:
        selected_steps = set(_csv_values(args.steps))
        invalid_steps = sorted(selected_steps - set(STEP_NAMES))
        if invalid_steps:
            raise ValueError(f"Unsupported steps: {', '.join(invalid_steps)}")
        for step_name in STEP_NAMES:
            setattr(config.steps, step_name, step_name in selected_steps)
    return config


def main():
    """Loads configuration and runs the enabled orchestration steps."""
    args = parse_args()
    setup_logger()
    config = apply_overrides(load_config(args.config_dir), args)
    context = PipelineContext(config, create_output_dir=not args.dry_run)
    
    orchestrator = Orchestrator(context)
    if args.dry_run:
        requested = orchestrator.get_steps_to_run()
        order = orchestrator.resolve_execution_order(requested)
        print("Resolved execution order:")
        for step in order:
            marker = "requested" if step.name in requested else "dependency"
            print(f"- {step.name} ({marker})")
        return
    orchestrator.run()


if __name__ == "__main__":
    main()
