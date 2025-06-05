"""
Token optimizer for chord system optimization.

This module orchestrates the pipeline (only token scoring and context
building are working for now).
"""

import argparse
import logging

from src.assignment.chord_assignment import assign_tokens_to_chords
from src.chord_generation.chord_generator import generate_chords
from src.common.config import GeneratorConfig
from src.token_generation.corpus_generator import generate_corpus
from src.token_generation.token_selection import extract_and_select_tokens_iteratively

logger = logging.getLogger(__name__)


def run_pipeline_stage(config: GeneratorConfig, stage: str):
    """Run a specific stage of the optimization pipeline

    Args:
        config: The generator configuration
        stage: Name of the pipeline stage to run
    """
    if stage == "corpus_generation":
        logger.info("Generating corpus...")
        generate_corpus(config)
    elif stage == "token_extraction":
        logger.info("Extracting, scoring and ordering tokens...")
        extract_and_select_tokens_iteratively(config)
    elif stage == "chords_generation":
        logger.info("Generating chords...")
        generate_chords(config)
    elif stage == "assignment":
        logger.info("Optimizing token-chord assignments...")
        assign_tokens_to_chords(config)
    elif stage == "analysis":
        logger.info("Analyzing assignments...")
        # Implementation for analysis will go here
        pass
    else:
        logger.error(f"Unknown pipeline stage: {stage}")


def main():
    """Run optimization from command line"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Optimize token selection for chord assignment"
    )

    # Add specific pipeline stages
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "corpus_generation",
            "token_extraction",
            "chords_generation",
            "assignment",
            "analysis",
        ],
        help="Pipeline stages to run",
    )

    # Add algorithm selection
    parser.add_argument(
        "--algorithm",
        choices=["greedy", "genetic", "compare"],
        help="Selection algorithm to use (or compare both)",
    )

    # Add configuration path
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )

    # Add verbosity option
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Add parallel processing override
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )

    args = parser.parse_args()

    # Load configuration
    config = GeneratorConfig.load_config(args.config)

    # Override configuration with command line arguments if provided
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled via command line")
    else:
        config.setup_logging()

    if args.algorithm:
        config.chord_assignment.algorithm = args.algorithm
        logger.debug(f"Using algorithm from command line argument: {args.algorithm}")

    if args.no_parallel:
        config.general.use_parallel_processing = False
        logger.debug("Parallel processing disabled via command line")

    # Determine pipeline stages to run
    stages = []
    if args.stages:
        stages = args.stages
    else:
        # Default to running all
        stages = [
            "corpus_generation",
            "token_extraction",
            "chords_generation",
            "assignment",
            "analysis",
        ]

    logger.info(f"Running pipeline stages: {', '.join(stages)}")

    # Run requested stages
    for stage in stages:
        run_pipeline_stage(config, stage)

    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()
