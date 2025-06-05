"""
Chord assignment module for optimizing token-chord assignments.

This module assigns selected tokens to chords by optimizing the total cost
(sum of usage_count * chord_usage_cost for all assignments).
"""

import logging
import os
from typing import List, Tuple

from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.shared_types import (
    Assignment,
    AssignmentSet,
    ChordCollection,
    ChordData,
    SetData,
    TokenCollection,
    TokenData,
    TokenType,
)

logger = logging.getLogger(__name__)


def filter_assignable_tokens(tokens: List[TokenData]) -> List[TokenData]:
    """Filter tokens to only include those that should be assigned to chords.

    Excludes single character tokens and sorts by usage_count in descending order.

    Args:
        tokens: List of all tokens

    Returns:
        List of tokens eligible for chord assignment, sorted by usage_count
    """
    # Filter out single character tokens
    assignable_tokens = [
        token
        for token in tokens
        if token.selected and token.token_type != TokenType.SINGLE_CHARACTER
    ]

    # Sort by usage_count in descending order (most used first)
    assignable_tokens.sort(key=lambda t: t.usage_count, reverse=True)

    logger.info(
        f"Found {len(assignable_tokens)} assignable tokens (excluding single characters)"
    )

    return assignable_tokens


def calculate_assignment_cost(token: TokenData, chord: ChordData) -> float:
    """Calculate the cost of assigning a token to a chord.

    Cost = token.usage_count * chord.usage_cost

    Args:
        token: The token to assign
        chord: The chord to assign to

    Returns:
        The assignment cost
    """
    return token.usage_count * chord.usage_cost


def assign_tokens_to_chords_greedy(
    tokens: List[TokenData], chords: List[ChordData]
) -> List[Assignment]:
    """Assign tokens to chords using a greedy algorithm.

    For each token (in order of decreasing usage_count), assigns it to the
    chord with the lowest usage_cost that hasn't been used yet.

    Args:
        tokens: List of tokens to assign (should be pre-sorted by usage_count)
        chords: List of available chords

    Returns:
        List of Assignment objects
    """
    assignments = []
    used_chords = set()

    # Sort chords by usage_cost (lowest first)
    sorted_chords = sorted(chords, key=lambda c: c.usage_cost)

    logger.info(
        f"Starting greedy assignment of {len(tokens)} tokens to {len(chords)} chords"
    )

    for token in tokens:
        # Find the lowest cost chord that hasn't been used
        assigned_chord = None
        for chord in sorted_chords:
            chord_key = id(chord)  # Use object id as unique identifier
            if chord_key not in used_chords:
                assigned_chord = chord
                used_chords.add(chord_key)
                break

        if assigned_chord is None:
            logger.warning(
                f"No available chord for token '{token.lower}' - ran out of chords"
            )
            break

        # Create assignment
        assignment = Assignment(token=token, chord=assigned_chord, metrics=None)
        assignments.append(assignment)

        cost = calculate_assignment_cost(token, assigned_chord)
        logger.debug(
            f"Assigned '{token.lower}' (usage: {token.usage_count}) to chord "
            f"with cost {assigned_chord.usage_cost:.4f} (total cost: {cost:.4f})"
        )

    logger.info(f"Completed greedy assignment: {len(assignments)} assignments created")
    return assignments


def calculate_total_assignment_cost(assignments: List[Assignment]) -> float:
    """Calculate the total cost of all assignments.

    Args:
        assignments: List of assignments

    Returns:
        Total cost (sum of usage_count * chord_usage_cost for all assignments)
    """
    total_cost = sum(
        calculate_assignment_cost(assignment.token, assignment.chord)
        for assignment in assignments
    )
    return total_cost


def assign_tokens_to_chords(config: GeneratorConfig) -> None:
    """Main function to assign tokens to chords and save results.

    Args:
        config: Generator configuration
    """
    # Initialize benchmarking if enabled
    benchmark = Benchmark(config.benchmark)

    logger.info("Starting token-chord assignment")
    benchmark.start_phase(BenchmarkPhase.WORD_ASSIGNMENT)

    # Load token collection
    token_filename = f"{os.path.splitext(config.active_corpus_file)[0]}_tokens_{config.token_analysis.top_n_tokens}.json"
    token_path = config.paths.tokens_dir / token_filename

    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_path}")

    logger.info(f"Loading tokens from {token_path}")
    token_collection = TokenCollection.load_from_file(token_path)

    # Load chord collection
    layout_name = os.path.splitext(config.active_layout_file)[0]
    chord_filename = f"{layout_name}_chords_{config.chord_generation.min_letter_count}_{config.chord_generation.max_letter_count}.json"
    chord_path = config.paths.chords_dir / chord_filename

    if not chord_path.exists():
        raise FileNotFoundError(f"Chord file not found: {chord_path}")

    logger.info(f"Loading chords from {chord_path}")
    chord_collection = ChordCollection.load_from_file(chord_path)

    # Filter tokens for assignment
    assignable_tokens = filter_assignable_tokens(token_collection.tokens)

    # Check if we have enough chords
    if len(assignable_tokens) > len(chord_collection.chords):
        logger.warning(
            f"More assignable tokens ({len(assignable_tokens)}) than available chords ({len(chord_collection.chords)}). "
            f"Some tokens will not be assigned."
        )

    # Limit to the number of chords we want to assign
    tokens_to_assign = assignable_tokens[: config.general.chords_to_assign]
    logger.info(f"Assigning {len(tokens_to_assign)} tokens to chords")

    # Perform assignment based on configured algorithm
    if config.chord_assignment.algorithm == "greedy":
        assignments = assign_tokens_to_chords_greedy(
            tokens_to_assign, chord_collection.chords
        )
    else:
        raise ValueError(
            f"Unknown assignment algorithm: {config.chord_assignment.algorithm}"
        )

    benchmark.update_phase(len(assignments))

    # Create assignment set
    total_cost = calculate_total_assignment_cost(assignments)

    # Calculate some basic metrics
    metrics = {
        "total_assignment_cost": total_cost,
        "num_assignments": len(assignments),
        "average_assignment_cost": (
            total_cost / len(assignments) if assignments else 0.0
        ),
        "algorithm_used": config.chord_assignment.algorithm,
    }

    # Create name based on collections and algorithm
    token_name = os.path.splitext(config.active_corpus_file)[0]
    chord_name = os.path.splitext(config.active_layout_file)[0]
    name = f"{token_name}_to_{chord_name}_{config.chord_assignment.algorithm}"

    assignment_set = AssignmentSet(name=name, assignments=assignments, metrics=metrics)

    # Log results
    total_cost = assignment_set.metrics["total_assignment_cost"]
    avg_cost = assignment_set.metrics["average_assignment_cost"]
    logger.info(f"Assignment completed:")
    logger.info(f"  Total cost: {total_cost:.4f}")
    logger.info(f"  Average cost per assignment: {avg_cost:.4f}")
    logger.info(f"  Number of assignments: {len(assignments)}")

    # Log top 5 assignments by cost
    assignments_by_cost = sorted(
        assignments,
        key=lambda a: calculate_assignment_cost(a.token, a.chord),
        reverse=True,
    )
    logger.info("Top 5 most expensive assignments:")
    for i, assignment in enumerate(assignments_by_cost[:5]):
        cost = calculate_assignment_cost(assignment.token, assignment.chord)
        logger.info(
            f"  {i+1}. '{assignment.token.lower}' (usage: {assignment.token.usage_count}) -> "
            f"chord cost {assignment.chord.usage_cost:.4f} (total: {cost:.4f})"
        )

    benchmark.end_phase()

    # Create SetData and save to file
    logger.info("Creating SetData and saving results")
    benchmark.start_phase(BenchmarkPhase.WRITING_OUTPUT)

    set_data = SetData(
        assignment_set=assignment_set,
        chord_collection=chord_collection,
        token_collection=token_collection,
    )

    # Create output filename
    output_filename = f"{assignment_set.name}_setdata.json"
    output_path = config.paths.results_dir / output_filename

    # Save SetData to JSON
    logger.info(f"Saving SetData to {output_path}")
    set_data.save_to_file(output_path)

    benchmark.end_phase()

    # Log benchmark results if enabled
    if config.benchmark.enabled:
        results = benchmark.get_results()
        logger.info(f"Benchmark results: {results}")

    logger.info("Token-chord assignment completed successfully")
