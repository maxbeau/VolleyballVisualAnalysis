"""
Main entrypoint for the Volleyball Visual Analysis orchestration.

This script provides a unified command-line interface to run the entire
analysis workflow, from initial object detection to final video overlay.

Examples:
    # Run the entire orchestration from start to finish
    python run_pipeline.py
"""
from core.context import PipelineContext
from core.logger import setup_logger
from core.orchestrator import Orchestrator


def main():
    """Loads configuration and runs the enabled orchestration steps."""
    setup_logger()
    context = PipelineContext.from_settings()
    
    orchestrator = Orchestrator(context)
    orchestrator.run()


if __name__ == "__main__":
    main()