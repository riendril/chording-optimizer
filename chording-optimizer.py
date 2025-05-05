"""
Token optimizer for chord system optimization.

This module orchestrates the optimization pipeline for chord-based text input systems.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set

from src.analyzing.token_scoring import (
    get_score_recalculation_function,
    score_token_collection,
)
from src.common.config import GeneratorConfig
from src.common.layout import load_keyboard_layout
from src.common.shared_types import TokenCollection
from src.token_generation.corpus_generator import generate_corpus
from src.token_generation.token_context import add_context_to_file
from src.token_generation.token_extraction import create_and_save_token_collection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TokenOptimizer:
    """Main class for token optimization process"""

    def __init__(self, config: GeneratorConfig):
        """Initialize token optimizer with configuration"""
        self.config = config
        self.layout_config = None
        self._load_layout()

    def _load_layout(self):
        """Load keyboard layout configuration"""
        layout_path = self.config.paths.get_layout_path(self.config.active_layout_file)
        self.layout_config = load_keyboard_layout(layout_path)
        logger.info(f"Loaded keyboard layout: {self.config.active_layout_file}")

    def progress_callback(self, step: int, message: str):
        """Progress update callback for long-running operations"""
        logger.info(f"Progress {step}: {message}")

    def run_pipeline_stage(self, stage: str):
        """Run a specific stage of the optimization pipeline"""
        if stage == "corpus_generation":
            logger.info("Generating corpus...")
            generate_corpus(self.config)
        elif stage == "token_extraction":
            logger.info("Extracting tokens...")
            corpus_path = self.config.paths.get_corpus_path(
                self.config.active_corpus_file
            )
            output_path = (
                self.config.paths.tokens_dir
                / f"{corpus_path.stem}_tokens_{self.config.token_analysis.top_n_tokens}.json"
            )
            create_and_save_token_collection(corpus_path, output_path, self.config)
        elif stage == "token_context":
            logger.info("Adding context to tokens...")
            tokens_path = self.config.paths.get_tokens_path(
                self.config.active_tokens_file
            )
            corpus_path = self.config.paths.get_corpus_path(
                self.config.active_corpus_file
            )
            output_path = (
                self.config.paths.tokens_dir / f"{tokens_path.stem}_with_context.json"
            )
            add_context_to_file(tokens_path, corpus_path, output_path)
        elif stage == "token_scoring":
            logger.info("Scoring tokens...")
            tokens_path = self.config.paths.get_tokens_path(
                self.config.active_tokens_file
            )
            token_collection = TokenCollection.load_from_file(tokens_path)
            score_token_collection(token_collection, self.layout_config, self.config)
        elif stage == "chords_generation":
            logger.info("Generating chords...")
            generate_chords_and_costs(self.config)
        elif stage == "assignment":
            logger.info("Optimizing token-chord assignments...")
            self.optimize_tokens(self.config.chord_assignment.algorithm)
        elif stage == "analysis":
            logger.info("Analyzing assignments...")
            # Implementation for analysis would go here
            pass
        else:
            logger.error(f"Unknown pipeline stage: {stage}")

    def optimize_tokens(self, algorithm: str):
        """Optimize token selection using specified algorithm"""
        # Load token collection
        tokens_path = self.config.paths.get_tokens_path(self.config.active_tokens_file)
        token_collection = TokenCollection.load_from_file(tokens_path)

        # Score tokens
        logger.info("Scoring tokens...")
        scored_collection = score_token_collection(
            token_collection, self.layout_config, self.config
        )

        # Get context weight from config
        context_weight = getattr(self.config.chord_assignment, "context_weight", 0.2)

        # Create recalculation function
        recalculate_func = get_score_recalculation_function(
            scored_collection, context_weight
        )

        # Select tokens
        logger.info(f"Selecting tokens using {algorithm} algorithm...")
        max_tokens = self.config.token_analysis.top_n_tokens

        if algorithm == "compare":
            self._compare_algorithms(scored_collection, recalculate_func, max_tokens)
        else:
            self._run_algorithm(
                algorithm, scored_collection, recalculate_func, max_tokens
            )

    def _run_algorithm(
        self,
        algorithm: str,
        collection: TokenCollection,
        recalculate_func,
        max_tokens: int,
    ):
        """Run a single optimization algorithm"""
        algorithm_config = {}
        if algorithm == "genetic":
            # Use configuration parameters for genetic algorithm
            algorithm_config = {
                "population_size": 50,
                "generations": 100,
                "elite_count": 5,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
            }

        selected_tokens = select_tokens(
            collection,
            max_tokens,
            recalculate_func,
            algorithm=algorithm,
            algorithm_config=algorithm_config,
            progress_callback=self.progress_callback,
        )

        # Create output path
        tokens_name = self.config.active_tokens_file.split(".")[0]
        output_path = (
            self.config.paths.results_dir / f"{tokens_name}_optimized_{algorithm}.json"
        )

        # Save results
        save_selected_tokens(selected_tokens, collection, output_path, algorithm)
        logger.info(f"Selected {len(selected_tokens)} tokens")
        logger.info(f"Saved optimized token collection to {output_path}")

    def _compare_algorithms(
        self, collection: TokenCollection, recalculate_func, max_tokens: int
    ):
        """Compare greedy and genetic algorithms on the same data"""
        logger.info("Comparing greedy and genetic algorithms...")

        # Run greedy algorithm
        logger.info("Running greedy algorithm...")
        greedy_tokens = select_tokens(
            collection, max_tokens, recalculate_func, algorithm="greedy"
        )

        # Run genetic algorithm
        logger.info("Running genetic algorithm...")
        genetic_tokens = select_tokens(
            collection,
            max_tokens,
            recalculate_func,
            algorithm="genetic",
            algorithm_config={
                "population_size": 50,
                "generations": 50,
                "elite_count": 5,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
            },
        )

        # Compare results
        common_tokens = set(greedy_tokens).intersection(set(genetic_tokens))

        # Calculate total score for each approach
        token_map = {t.original: t for t in collection.tokens}
        greedy_score = sum(token_map[t].score for t in greedy_tokens if t in token_map)
        genetic_score = sum(
            token_map[t].score for t in genetic_tokens if t in token_map
        )

        # Log summary
        logger.info("\nAlgorithm Comparison:")
        logger.info(
            f"Greedy: {len(greedy_tokens)} tokens, total score: {greedy_score:.2f}"
        )
        logger.info(
            f"Genetic: {len(genetic_tokens)} tokens, total score: {genetic_score:.2f}"
        )
        logger.info(
            f"Common tokens: {len(common_tokens)} ({len(common_tokens)/len(genetic_tokens)*100:.2f}%)"
        )
        logger.info(
            f"Score improvement: {(genetic_score-greedy_score)/greedy_score*100:.2f}%"
        )


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
            "all",
        ],
        help="Pipeline stages to run",
    )

    # Add algorithm selection
    parser.add_argument(
        "--algorithm",
        choices=["greedy", "genetic", "compare"],
        default="greedy",
        help="Selection algorithm to use (or compare both)",
    )

    # Add verbosity option
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Set up logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = GeneratorConfig.load_config()

    # Create optimizer
    optimizer = TokenOptimizer(config)

    # Determine pipeline stages to run
    stages = []
    if args.stages:
        if "all" in args.stages:
            stages = [
                "corpus_generation",
                "token_extraction",
                "token_context",
                "token_scoring",
                "chords_generation",
                "assignment",
                "analysis",
            ]
        else:
            stages = args.stages
    else:
        # Default to just running assignment
        stages = ["assignment"]

    # Run requested stages
    for stage in stages:
        if stage == "assignment":
            optimizer.optimize_tokens(args.algorithm)
        else:
            optimizer.run_pipeline_stage(stage)


if __name__ == "__main__":
    main()
