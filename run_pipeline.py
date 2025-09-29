"""
Main entrypoint for the Volleyball Visual Analysis pipeline.

This script provides a unified command-line interface to run the entire
analysis workflow, from initial object detection to final video overlay.
It allows users to selectively run specific stages of the pipeline.

Examples:
    # Run the entire pipeline from start to finish
    python run_pipeline.py

    # Run only the detection steps
    python run_pipeline.py --steps detection

    # Run detection for all targets, then generate the final overlay
    python run_pipeline.py --steps detection,overlay
"""
import logging
from typing import List, Dict, Set

from core.context import PipelineContext
from core.logger import setup_logger
from core.steps import PipelineStep
from pipeline.steps import (
    DetectionStep,
    CourtProcessingStep,
    CourtHomographyStep,
    PlayersTrackingStep,
    TrajectoryAnalysisStep,
    OverlayStep,
)


def get_all_steps(context: PipelineContext) -> Dict[str, PipelineStep]:
    """Instantiates and returns all available pipeline steps."""
    steps = [
        DetectionStep(context, "court"),
        DetectionStep(context, "players"),
        DetectionStep(context, "ball"),
        DetectionStep(context, "actions"),
        CourtProcessingStep(context),
        CourtHomographyStep(context),
        PlayersTrackingStep(context),
        TrajectoryAnalysisStep(context),
        OverlayStep(context),
    ]
    return {step.name: step for step in steps}


def get_steps_to_run(context: PipelineContext) -> Set[str]:
    """Determines which steps to run based on the configuration."""
    steps_config = context.config.steps
    steps_to_run = set()

    # Map config flags to the steps they enable
    # The 'None' key represents steps that are always included if their flag is true
    step_groups = {
        "detection": ["detection_court", "detection_players", "detection_ball", "detection_actions"],
        "court_processing": [None],
        "court_homography": [None],
        "players_tracking": [None],
        "trajectory_analysis": [None],
        "overlay": [None],
    }

    for config_flag, step_names in step_groups.items():
        if getattr(steps_config, config_flag, False):
            if step_names == [None]:
                steps_to_run.add(config_flag)
            else:
                steps_to_run.update(step_names)
                
    return steps_to_run


def resolve_execution_order(
    steps: Dict[str, PipelineStep], steps_to_run: Set[str]
) -> List[PipelineStep]:
    """
    Resolves the execution order of steps based on their dependencies
    using topological sort.
    """
    resolved_order = []
    resolved_steps = set()
    
    # Include dependencies of the requested steps
    required_steps = set(steps_to_run)
    for step_name in steps_to_run:
        q = list(steps[step_name].dependencies)
        while q:
            dep_name = q.pop(0)
            if dep_name not in required_steps:
                required_steps.add(dep_name)
                q.extend(steps[dep_name].dependencies)

    while len(resolved_steps) < len(required_steps):
        ready_to_run = {
            name
            for name in required_steps
            if name not in resolved_steps
            and all(dep in resolved_steps for dep in steps[name].dependencies)
        }

        if not ready_to_run:
            raise RuntimeError("Circular dependency detected in pipeline steps.")

        # Sort for deterministic execution order
        for step_name in sorted(list(ready_to_run)):
            resolved_order.append(steps[step_name])
            resolved_steps.add(step_name)
            
    # The full resolved order should be returned to ensure dependencies are run
    return resolved_order


def main():
    """Loads configuration and runs the enabled pipeline steps."""
    setup_logger()
    context = PipelineContext.from_settings()
    all_steps = get_all_steps(context)
    steps_to_run_names = get_steps_to_run(context)
    
    try:
        execution_order = resolve_execution_order(all_steps, steps_to_run_names)
    except Exception as e:
        logging.error(f"❌ Failed to resolve pipeline execution order: {e}", exc_info=True)
        return

    logging.info("🚀 Starting Volleyball Analysis Pipeline...")
    logging.info(f"📹 Video Input: {context.video_path}")
    logging.info(f"📂 Output Dir: {context.output_dir}")
    logging.info(f"🏃 Steps to run: {[s.name for s in execution_order]}")

    for step in execution_order:
        try:
            step()  # Step instances are callable
        except Exception as e:
            logging.error(f"❌ Step '{step.name}' failed with an error: {e}", exc_info=True)
            logging.error("Aborting pipeline.")
            return

    logging.info("✅ Pipeline finished successfully.")


if __name__ == "__main__":
    main()