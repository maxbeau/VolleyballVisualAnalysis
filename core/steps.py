from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set

from core.context import PipelineContext


class PipelineStep(ABC):
    """Abstract base class for a step in the processing pipeline."""

    def __init__(self, context: PipelineContext):
        self.context = context
        self.settings = context.config

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

    @abstractmethod
    def run(self) -> None:
        """
        Executes the main logic of the pipeline step.
        This method should perform the processing and register any outputs
        as artifacts in the context.
        """
        pass

import logging

class PipelineStep(ABC):
    """Abstract base class for a step in the processing pipeline."""

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

    @abstractmethod
    def run(self) -> None:
        """
        Executes the main logic of the pipeline step.
        This method should perform the processing and register any outputs
        as artifacts in the context.
        """
        pass

    def __call__(self) -> None:
        """Allows the step instance to be called directly."""
        self.logger.info(f"--- Running step: {self.name} ---")
        self.run()
        self.logger.info(f"--- Step '{self.name}' finished ---")
