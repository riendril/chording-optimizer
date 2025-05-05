"""
Token optimizer for chord system optimization.

This module orchestrates the pipeline (only token scoring and context
building are working for now).
"""

import argparse
import logging

from src.analyzing.token_scoring import score_token_collection
from src.common.config import GeneratorConfig
from src.token_generation.corpus_generator import generate_corpus
from src.token_generation.token_context import add_context_to_file
from src.token_generation.token_extraction import extract_tokens

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
        logger.info("Extracting tokens...")
        extract_tokens(config)
    elif stage == "token_context":
        logger.info("Adding context to tokens...")
        add_context_to_file(config)
    elif stage == "token_scoring":
        logger.info("Scoring tokens...")
        score_token_collection(config)
    elif stage == "chords_generation":
        logger.info("Generating chords...")
        # Implementation for chord generation will go here
        pass
    elif stage == "assignment":
        logger.info("Optimizing token-chord assignments...")
        # Implementation for assignment will go here
        pass
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
            "token_context",
            "token_scoring",
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

    # Determine pipeline stages to run
    stages = []
    if args.stages:
        stages = args.stages
    else:
        # Default to running all
        stages = [
            "corpus_generation",
            "token_extraction",
            "token_context",
            "token_scoring",
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
