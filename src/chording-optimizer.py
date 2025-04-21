"""
Token optimizer for chord system optimization.

This module orchestrates the pipeline
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.analyzing.token_scoring import (
    get_score_recalculation_function,
    score_token_collection,
)
from src.assignment.token_selection import save_selected_tokens, select_tokens
from src.common.config import GeneratorConfig
from src.common.layout import load_keyboard_layout
from src.common.shared_types import ChordData, ContextInfo, TokenCollection, TokenData
from src.token_generation.token_extraction import (
    create_token_collection_with_context,
    extract_tokens_with_context,
    read_corpus_from_file,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TokenOptimizer:
    """Main class for token optimization process"""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize token optimizer with configuration.

        Args:
            config: Generator configuration
        """
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

    def optimize_from_corpus(
        self,
        corpus_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        algorithm: str = "greedy",
        max_tokens: Optional[int] = None,
        context_weight: float = 0.2,
        extract_context: bool = True,
    ) -> TokenCollection:
        """
        Complete optimization pipeline starting from a corpus file.

        Args:
            corpus_path: Path to corpus file (uses active corpus if None)
            output_path: Path to save results (auto-generated if None)
            algorithm: Selection algorithm ('greedy' or 'genetic')
            max_tokens: Maximum tokens to select (from config if None)
            context_weight: Weight for context-based adjustments
            extract_context: Whether to extract context information

        Returns:
            Optimized token collection
        """
        # Determine paths
        if corpus_path is None:
            corpus_path = self.config.paths.get_corpus_path(
                self.config.active_corpus_file
            )

        if output_path is None:
            corpus_name = corpus_path.stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.config.paths.tokens_dir
                / f"{corpus_name}_optimized_{algorithm}_{timestamp}.json"
            )

        if max_tokens is None:
            max_tokens = self.config.token_analysis.top_n_tokens

        # Load corpus
        logger.info(f"Loading corpus from {corpus_path}")
        corpus = read_corpus_from_file(corpus_path)

        # Extract tokens
        logger.info("Extracting tokens and context information...")
        tokens, context_info = extract_tokens_with_context(
            corpus,
            self.config,
            window_size=2 if extract_context else 0,
            min_freq=2,
            show_progress=True,
        )

        # Create token collection
        collection_name = f"{corpus_path.stem}_tokens_{len(tokens)}"
        token_collection = create_token_collection_with_context(
            tokens, collection_name, source=str(corpus_path), context_info=context_info
        )

        # Score tokens
        logger.info("Scoring tokens...")
        scored_collection = score_token_collection(
            token_collection, self.layout_config, self.config
        )

        # Create recalculation function
        recalculate_func = get_score_recalculation_function(
            scored_collection, context_weight
        )

        # Select tokens
        logger.info(f"Selecting tokens using {algorithm} algorithm...")
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
            scored_collection,
            max_tokens,
            recalculate_func,
            algorithm=algorithm,
            algorithm_config=algorithm_config,
            progress_callback=self.progress_callback,
        )

        # Create final collection
        logger.info(f"Selected {len(selected_tokens)} tokens")

        # Save results
        save_selected_tokens(selected_tokens, scored_collection, output_path, algorithm)

        logger.info(f"Saved optimized token collection to {output_path}")

        return scored_collection

    def optimize_from_tokens(
        self,
        tokens_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        algorithm: str = "greedy",
        max_tokens: Optional[int] = None,
        context_weight: float = 0.2,
    ) -> TokenCollection:
        """
        Optimize from existing token collection.

        Args:
            tokens_path: Path to token collection (uses active tokens if None)
            output_path: Path to save results (auto-generated if None)
            algorithm: Selection algorithm ('greedy' or 'genetic')
            max_tokens: Maximum tokens to select (from config if None)
            context_weight: Weight for context-based adjustments

        Returns:
            Optimized token collection
        """
        # Determine paths
        if tokens_path is None:
            tokens_path = self.config.paths.get_tokens_path(
                self.config.active_tokens_file
            )

        if output_path is None:
            tokens_name = tokens_path.stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.config.paths.tokens_dir
                / f"{tokens_name}_optimized_{algorithm}_{timestamp}.json"
            )

        if max_tokens is None:
            max_tokens = self.config.token_analysis.top_n_tokens

        # Load token collection
        logger.info(f"Loading token collection from {tokens_path}")
        token_collection = TokenCollection.load_from_file(tokens_path)

        # Score tokens
        logger.info("Scoring tokens...")
        scored_collection = score_token_collection(
            token_collection, self.layout_config, self.config
        )

        # Create recalculation function
        recalculate_func = get_score_recalculation_function(
            scored_collection, context_weight
        )

        # Select tokens
        logger.info(f"Selecting tokens using {algorithm} algorithm...")
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
            scored_collection,
            max_tokens,
            recalculate_func,
            algorithm=algorithm,
            algorithm_config=algorithm_config,
            progress_callback=self.progress_callback,
        )

        # Create final collection
        logger.info(f"Selected {len(selected_tokens)} tokens")

        # Save results
        save_selected_tokens(selected_tokens, scored_collection, output_path, algorithm)

        logger.info(f"Saved optimized token collection to {output_path}")

        return scored_collection

    def compare_algorithms(
        self,
        corpus_path: Optional[Path] = None,
        tokens_path: Optional[Path] = None,
        max_tokens: int = 100,
        context_weight: float = 0.2,
    ) -> Dict:
        """
        Compare greedy and genetic algorithms on the same data.

        Args:
            corpus_path: Path to corpus file (optional)
            tokens_path: Path to token collection (optional)
            max_tokens: Maximum tokens to select
            context_weight: Weight for context-based adjustments

        Returns:
            Dictionary with comparison results
        """
        # Determine input source
        if corpus_path:
            logger.info(f"Loading corpus from {corpus_path}")
            corpus = read_corpus_from_file(corpus_path)
            tokens, context_info = extract_tokens_with_context(
                corpus, self.config, window_size=2, min_freq=2
            )
            collection_name = f"{corpus_path.stem}_tokens_{len(tokens)}"
            token_collection = create_token_collection_with_context(
                tokens,
                collection_name,
                source=str(corpus_path),
                context_info=context_info,
            )
        elif tokens_path:
            logger.info(f"Loading token collection from {tokens_path}")
            token_collection = TokenCollection.load_from_file(tokens_path)
        else:
            tokens_path = self.config.paths.get_tokens_path(
                self.config.active_tokens_file
            )
            logger.info(f"Loading active token collection from {tokens_path}")
            token_collection = TokenCollection.load_from_file(tokens_path)

        # Score tokens
        scored_collection = score_token_collection(
            token_collection, self.layout_config, self.config
        )

        # Create recalculation function
        recalculate_func = get_score_recalculation_function(
            scored_collection, context_weight
        )

        # Run greedy algorithm
        logger.info("Running greedy algorithm...")
        start_time = time.time()
        greedy_tokens = select_tokens(
            scored_collection, max_tokens, recalculate_func, algorithm="greedy"
        )
        greedy_time = time.time() - start_time

        # Run genetic algorithm
        logger.info("Running genetic algorithm...")
        start_time = time.time()
        genetic_tokens = select_tokens(
            scored_collection,
            max_tokens,
            recalculate_func,
            algorithm="genetic",
            algorithm_config={
                "population_size": 50,
                "generations": 50,  # Fewer generations for comparison
                "elite_count": 5,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
            },
        )
        genetic_time = time.time() - start_time

        # Compare results
        common_tokens = set(greedy_tokens).intersection(set(genetic_tokens))

        # Calculate total score for each approach
        token_map = {t.original: t for t in scored_collection.tokens}
        greedy_score = sum(token_map[t].score for t in greedy_tokens if t in token_map)
        genetic_score = sum(
            token_map[t].score for t in genetic_tokens if t in token_map
        )

        # Prepare comparison results
        comparison = {
            "greedy": {
                "tokens": greedy_tokens,
                "score": greedy_score,
                "runtime": greedy_time,
            },
            "genetic": {
                "tokens": genetic_tokens,
                "score": genetic_score,
                "runtime": genetic_time,
            },
            "comparison": {
                "common_tokens": len(common_tokens),
                "overlap_percentage": len(common_tokens) / len(genetic_tokens) * 100,
                "score_difference": genetic_score - greedy_score,
                "score_improvement": (
                    ((genetic_score - greedy_score) / greedy_score * 100)
                    if greedy_score > 0
                    else 0
                ),
            },
        }

        # Log summary
        logger.info("\nAlgorithm Comparison:")
        logger.info(
            f"Greedy total score: {greedy_score:.2f} (runtime: {greedy_time:.2f}s)"
        )
        logger.info(
            f"Genetic total score: {genetic_score:.2f} (runtime: {genetic_time:.2f}s)"
        )
        logger.info(
            f"Common tokens: {len(common_tokens)} ({comparison['comparison']['overlap_percentage']:.2f}%)"
        )
        logger.info(
            f"Score difference: {comparison['comparison']['score_difference']:.2f} "
            f"({comparison['comparison']['score_improvement']:.2f}%)"
        )

        return comparison


def run_optimization(config_path: Optional[Path] = None):
    """
    Run optimization from command line.

    Args:
        config_path: Path to config file
    """
    # Load configuration
    config = GeneratorConfig.load_config(config_path)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Optimize token selection for chord assignment"
    )
    parser.add_argument("--corpus", type=Path, help="Path to corpus file")
    parser.add_argument("--tokens", type=Path, help="Path to token collection file")
    parser.add_argument("--output", type=Path, help="Path to output file")
    parser.add_argument(
        "--algorithm",
        choices=["greedy", "genetic"],
        default="greedy",
        help="Selection algorithm to use",
    )
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to select")
    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.2,
        help="Weight for context-based adjustments",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare greedy and genetic algorithms"
    )
    parser.add_argument(
        "--no-context", action="store_true", help="Disable context extraction"
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = TokenOptimizer(config)

    # Run requested operation
    if args.compare:
        optimizer.compare_algorithms(
            corpus_path=args.corpus,
            tokens_path=args.tokens,
            max_tokens=args.max_tokens or 100,
            context_weight=args.context_weight,
        )
    elif args.corpus:
        optimizer.optimize_from_corpus(
            corpus_path=args.corpus,
            output_path=args.output,
            algorithm=args.algorithm,
            max_tokens=args.max_tokens,
            context_weight=args.context_weight,
            extract_context=not args.no_context,
        )
    elif args.tokens:
        optimizer.optimize_from_tokens(
            tokens_path=args.tokens,
            output_path=args.output,
            algorithm=args.algorithm,
            max_tokens=args.max_tokens,
            context_weight=args.context_weight,
        )
    else:
        # Use active corpus from config
        optimizer.optimize_from_corpus(
            output_path=args.output,
            algorithm=args.algorithm,
            max_tokens=args.max_tokens,
            context_weight=args.context_weight,
            extract_context=not args.no_context,
        )


if __name__ == "__main__":
    run_optimization()
