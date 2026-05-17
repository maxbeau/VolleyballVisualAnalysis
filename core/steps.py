from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set
import logging

from core.context import PipelineContext


class OrchestrationStep(ABC):
    """Abstract base class for a step in the processing orchestration."""

    def __init__(self, context: PipelineContext):
        self.context = context
        self.settings = context.config
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """A unique identifier for the step."""
        pass

    @property
    def dependencies(self) -> Set[str]:
        """
        A set of step names that must be completed before this step can run.
        Defaults to an empty set (no dependencies).
        """
        return set()

    @property
    def output_artifacts(self) -> Dict[str, str | Path]:
        """Artifacts produced by this step, keyed by context artifact name."""
        return {}

    def restore_outputs(self) -> bool:
        """Register existing output files so dependency steps can be reused."""
        artifacts = self.output_artifacts
        if not artifacts:
            return False

        missing = [
            self.context.output_dir / Path(relative_path)
            for relative_path in artifacts.values()
            if not (self.context.output_dir / Path(relative_path)).exists()
        ]
        if missing:
            return False

        for artifact_name, relative_path in artifacts.items():
            self.context.register_artifact(artifact_name, relative_path)
        self.logger.info("Reusing existing artifacts for step '%s'.", self.name)
        return True

    @abstractmethod
    def run(self) -> None:
        """
        Executes the main logic of the orchestration step.
        This method should perform the processing and register any outputs
        as artifacts in the context.
        """
        pass

    def __call__(self) -> None:
        """Allows the step instance to be called directly."""
        self.logger.info(f"--- Running step: {self.name} ---")
        self.run()
        self.logger.info(f"--- Step '{self.name}' finished ---")
