"""
Core orchestration logic for the Volleyball Visual Analysis project.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Set

from core.context import PipelineContext
from core.steps import OrchestrationStep
from orchestration.steps import (
    DetectionStep,
    CourtProcessingStep,
    CourtHomographyStep,
    PlayersTrackingStep,
    TrajectoryAnalysisStep,
    OverlayStep,
)


DETECTION_TARGETS = ("court", "players", "ball", "actions", "net")


class Orchestrator:
    """Manages the setup and execution of orchestration steps."""

    def __init__(self, context: PipelineContext):
        self.context = context
        self.all_steps = self._get_all_steps()

    def _get_all_steps(self) -> Dict[str, OrchestrationStep]:
        """Instantiates and returns all available orchestration steps."""
        steps = [
            DetectionStep(self.context, "court"),
            DetectionStep(self.context, "players"),
            DetectionStep(self.context, "ball"),
            DetectionStep(self.context, "actions"),
            DetectionStep(self.context, "net"),
            CourtProcessingStep(self.context),
            CourtHomographyStep(self.context),
            PlayersTrackingStep(self.context),
            TrajectoryAnalysisStep(self.context),
            OverlayStep(self.context),
        ]
        return {step.name: step for step in steps}

    def get_steps_to_run(self) -> Set[str]:
        """Determines which steps to run based on the configuration."""
        steps_config = self.context.config.steps
        steps_to_run = set()
        detection_targets = self._configured_detection_targets()

        step_groups = {
            "detection": [f"detection_{target}" for target in detection_targets],
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

    def _configured_detection_targets(self) -> List[str]:
        """Return detection targets selected for an explicit detection run."""
        detection_config = self.context.config.detection
        if detection_config is None:
            return ["court", "players", "ball"]
        return [target for target in detection_config.targets if target in DETECTION_TARGETS]

    def resolve_execution_order(self, steps_to_run: Set[str]) -> List[OrchestrationStep]:
        """
        Resolves the execution order of steps based on their dependencies
        using topological sort.
        """
        resolved_order = []
        resolved_steps = set()
        
        required_steps = set(steps_to_run)
        for step_name in steps_to_run:
            q = list(self.all_steps[step_name].dependencies)
            while q:
                dep_name = q.pop(0)
                if dep_name not in required_steps:
                    required_steps.add(dep_name)
                    q.extend(self.all_steps[dep_name].dependencies)

        while len(resolved_steps) < len(required_steps):
            ready_to_run = {
                name
                for name in required_steps
                if name not in resolved_steps
                and all(dep in resolved_steps for dep in self.all_steps[name].dependencies)
            }

            if not ready_to_run:
                raise RuntimeError("Circular dependency detected in orchestration steps.")

            for step_name in sorted(list(ready_to_run)):
                resolved_order.append(self.all_steps[step_name])
                resolved_steps.add(step_name)
                
        return resolved_order

    def run(self):
        """Resolves dependencies and runs the configured orchestration."""
        steps_to_run_names = self.get_steps_to_run()
        
        try:
            execution_order = self.resolve_execution_order(steps_to_run_names)
        except Exception as e:
            logging.error(f"❌ Failed to resolve orchestration execution order: {e}", exc_info=True)
            return

        logging.info("🚀 Starting Volleyball Analysis Orchestration...")
        logging.info(f"📹 Video Input: {self.context.video_path}")
        logging.info(f"📂 Output Dir: {self.context.output_dir}")
        logging.info(f"🏃 Steps to run: {[s.name for s in execution_order]}")

        for step in execution_order:
            try:
                is_explicit_step = step.name in steps_to_run_names
                if (
                    self.context.config.global_settings.reuse_artifacts
                    and not is_explicit_step
                    and step.restore_outputs()
                ):
                    continue
                step()  # Step instances are callable
            except Exception as e:
                logging.error(f"❌ Step '{step.name}' failed with an error: {e}", exc_info=True)
                logging.error("Aborting orchestration.")
                return

        logging.info("✅ Orchestration finished successfully.")
