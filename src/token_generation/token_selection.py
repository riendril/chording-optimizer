"""
Token selection module with iterative selection algorithm.

This module processes corpus text to:
1. Initialize a list of selected tokens (starting with single characters)
2. Iteratively find optimal segmentation of text using currently selected tokens
3. Generate new token candidates from the optimally segmented text
4. Score all current tokens based on discomfort (per-use cost)
5. Select next eligible token based on candidate_score (discomfort * frequency)
6. Output a ranked list of tokens
"""

import logging
import os
from collections import Counter
from typing import Dict, Optional

import tqdm

from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData, TokenType
from src.token_generation.text_segmentation import (
    find_optimal_text_segmentation_in_chunks,
    visualize_text_segmentation,
)
from src.token_generation.token_extraction import (
    extract_tokens_from_segmentation,
    extract_tokens_from_segmentation_parallel,
    extract_words_from_text,
    set_word_set_for_cache,
)
from src.token_generation.token_scoring import (
    calculate_candidate_score,
    calculate_discomfort_score,
)

logger = logging.getLogger(__name__)


def select_tokens_iteratively(
    text: str,
    min_token_length: int,
    max_token_length: int,
    learning_limit: TokenType,
    chords_to_assign: int,
    top_n_tokens: int,
    word_set_id: int,
    benchmark: Benchmark,
    debug_options: dict,
    layout_comfort: Optional[Dict[str, float]] = None,
    use_parallel: bool = False,
) -> TokenCollection:
    """Select tokens iteratively by finding optimal segmentation after each selection.

    Args:
        text: Input corpus text
        min_token_length: Minimum token length
        max_token_length: Maximum token length
        learning_limit: Maximum token type complexity
        chords_to_assign: Number of tokens to select
        top_n_tokens: Number of top tokens to include in output
        word_set_id: Identifier for word set
        benchmark: Benchmark instance
        debug_options: Debugging options
        layout_comfort: Optional dict mapping characters to comfort scores
        use_parallel: Whether to use parallel processing

    Returns:
        TokenCollection with selected tokens and top candidates
    """
    # Initialize with single letter tokens
    selected_tokens = []

    # Store all candidate tokens ever seen
    all_candidate_tokens = {}

    # Text length for normalization
    text_length = len(text)

    # Create all single letter tokens
    unique_chars = set(text.lower())
    logger.info(f"Found {len(unique_chars)} unique characters in corpus")

    for char in unique_chars:
        token_data = TokenData(
            lower=char,
            length=1,
            token_type=TokenType.SINGLE_CHARACTER,
            text_count=text.lower().count(char),
            usage_count=text.lower().count(char),
            rank=0,
            score=0.0,  # Will store discomfort score
            selected=True,
            best_current_combination=[char],
        )
        # Calculate discomfort score (stored in score field)
        token_data.score = calculate_discomfort_score(token_data, layout_comfort)

        selected_tokens.append(token_data)
        all_candidate_tokens[char] = token_data

    # Sort by candidate score and assign ranks
    selected_tokens.sort(
        key=lambda t: calculate_candidate_score(t, text_length), reverse=True
    )

    for i, token in enumerate(selected_tokens):
        token.rank = i + 1

    contained_character_count = len(selected_tokens)
    logger.info(f"Initialized with {contained_character_count} single character tokens")

    # Main selection loop
    progress_bar = tqdm.tqdm(
        total=chords_to_assign,
        desc="Selecting tokens iteratively",
    )

    # Keep track of iteration for debugging
    iteration = 0

    while len(selected_tokens) - contained_character_count < chords_to_assign:
        iteration += 1

        # Find optimal segmentation using current selected tokens
        segmentation = find_optimal_text_segmentation_in_chunks(text, selected_tokens)

        # Debug: Visualize segmentation if enabled
        if debug_options.get("print_segmentation", False):
            visualization = visualize_text_segmentation(
                text, segmentation, segments_to_show=1
            )
            logger.info(
                f"Segmentation visualization (iteration {iteration}):\n{visualization}"
            )

        # Extract new token candidates
        if use_parallel:
            candidate_tokens = extract_tokens_from_segmentation_parallel(
                segmentation, text, min_token_length, max_token_length, word_set_id
            )
        else:
            candidate_tokens = extract_tokens_from_segmentation(
                segmentation, text, min_token_length, max_token_length, word_set_id
            )

        if not candidate_tokens:
            logger.warning("No more token candidates available")
            break

        # Calculate discomfort and candidate scores for all candidates
        for candidate in candidate_tokens:
            # Calculate discomfort score (stored in score field)
            candidate.score = calculate_discomfort_score(candidate, layout_comfort)

            # Add to all_candidate_tokens dict
            if candidate.lower not in all_candidate_tokens:
                all_candidate_tokens[candidate.lower] = candidate
            else:
                # Update existing candidate if better
                existing = all_candidate_tokens[candidate.lower]
                existing.text_count = candidate.text_count
                existing.usage_count = candidate.usage_count

                # Keep the better combination (shorter is better)
                if len(candidate.best_current_combination) < len(
                    existing.best_current_combination
                ):
                    existing.best_current_combination = (
                        candidate.best_current_combination
                    )

                # Keep discomfort score calculation consistent
                existing.score = calculate_discomfort_score(existing, layout_comfort)

        # Sort candidates by candidate score (not discomfort score)
        candidate_tokens.sort(
            key=lambda t: calculate_candidate_score(t, text_length), reverse=True
        )

        # Debug: show top candidates if enabled
        if debug_options.get("print_candidates", False):
            top_candidates = (
                candidate_tokens[:10]
                if len(candidate_tokens) > 10
                else candidate_tokens
            )
            candidate_info = "\n".join(
                f"  {i+1}. '{c.lower}' (discomfort: {c.score:.4f}, "
                f"candidate_score: {calculate_candidate_score(c, text_length):.6f}, "
                f"count: {c.text_count}, type: {c.token_type.name})"
                for i, c in enumerate(top_candidates)
            )
            logger.info(f"Top candidates (iteration {iteration}):\n{candidate_info}")

        next_token = None

        # Filter eligible candidates based on type, character length, and uniqueness
        for candidate in candidate_tokens:
            if (
                candidate.token_type <= learning_limit
                and len(candidate.lower) <= max_token_length
                and not any(
                    selected.lower == candidate.lower for selected in selected_tokens
                )
            ):
                next_token = candidate
                break

        if not next_token:
            logger.warning("No more eligible tokens available")
            break

        # Add to selected tokens
        next_token.selected = True
        next_token.best_current_combination = [
            next_token.lower
        ]  # Set its own representation
        selected_tokens.append(next_token)

        # Log with both discomfort and candidate score
        logger.info(
            f"Selected token: '{next_token.lower}' (discomfort: {next_token.score:.4f}, "
            f"candidate_score: {calculate_candidate_score(next_token, text_length):.6f}, "
            f"count: {next_token.text_count}, type: {next_token.token_type.name})"
        )

        # Update progress
        progress_bar.update(1)
        benchmark.update_phase(len(selected_tokens))

    progress_bar.close()

    # Perform one final segmentation after all tokens have been selected
    logger.info("Performing final segmentation with all selected tokens")
    final_segmentation = find_optimal_text_segmentation_in_chunks(text, selected_tokens)

    # Recalculate token frequencies based on final segmentation
    token_usage = Counter()
    for token_text, _, _, _ in final_segmentation:
        token_usage[token_text.lower()] += 1

    # Update token counts and recalculate scores
    for token in selected_tokens:
        if token.lower in token_usage:
            token.text_count = token_usage[token.lower]
            token.usage_count = token_usage[token.lower]

    # Visualize final segmentation if enabled
    if debug_options.get("print_segmentation", False):
        visualization = visualize_text_segmentation(
            text, final_segmentation, segments_to_show=3
        )
        logger.info(f"Final segmentation visualization:\n{visualization}")
    else:
        # Always show at least one visualization sample of the final segmentation
        visualization = visualize_text_segmentation(text, final_segmentation)
        logger.info(f"Sample of final segmentation:\n{visualization}")

    # Prepare final token collection including both selected tokens and top candidates
    final_tokens = []

    # First include all selected tokens
    final_tokens.extend(selected_tokens)

    # Convert all_candidate_tokens dict to list
    all_candidates_list = list(all_candidate_tokens.values())

    # Sort all candidates by candidate score
    all_candidates_list.sort(
        key=lambda t: calculate_candidate_score(t, text_length), reverse=True
    )

    # Add non-selected top tokens to reach top_n_tokens
    non_selected_top_tokens = [
        t
        for t in all_candidates_list
        if not t.selected and t.token_type <= learning_limit
    ][: top_n_tokens - len(selected_tokens)]

    final_tokens.extend(non_selected_top_tokens)

    # Sort all tokens by selected status and candidate score
    final_tokens.sort(
        key=lambda t: (
            -1 if t.selected else 0,
            calculate_candidate_score(t, text_length),
        ),
        reverse=True,
    )

    # Update ranks
    for i, token in enumerate(final_tokens):
        token.rank = i + 1

    # Create token collection with final tokens
    token_collection = TokenCollection(
        name="iterative_selection",
        tokens=final_tokens,
        ordered_by_frequency=True,
        source="iterative_approach",
    )

    return token_collection


def extract_and_select_tokens_iteratively(config: GeneratorConfig) -> None:
    """Extract, score, and order tokens from the corpus using the iterative approach.

    Args:
        config: Generator configuration
    """
    # Initialize benchmarking if enabled
    benchmark = Benchmark(config.benchmark)

    logger.info("Starting token extraction with iterative approach")

    # Get corpus file path
    corpus_path = config.paths.get_corpus_path(config.active_corpus_file)

    # Load corpus text
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    # Extract words for token classification
    logger.info("Extracting words from corpus using word boundaries")
    word_set = extract_words_from_text(corpus_text)
    logger.info(f"Extracted {len(word_set)} unique words from corpus")

    # Set global word set for caching and get identifier
    word_set_id = set_word_set_for_cache(word_set)

    # Set up debug options based on config
    debug_options = {
        "print_segmentation": config.debug.enabled and config.debug.print_cost_details,
        "print_candidates": config.debug.enabled and config.debug.print_cost_details,
        "save_intermediate": config.debug.enabled
        and config.debug.save_intermediate_results,
    }

    # Load keyboard layout comfort information if available
    layout_comfort = None
    try:
        # Get active layout file
        from src.common.layout import load_keyboard_layout

        layout_path = config.paths.get_layout_path(config.active_layout_file)
        layout_data = load_keyboard_layout(layout_path)

        if "comfort" in layout_data:
            layout_comfort = layout_data["comfort"]
            logger.info(
                f"Loaded comfort information from keyboard layout '{config.active_layout_file}'"
            )
    except Exception as e:
        logger.warning(f"Could not load layout comfort information: {e}")
        logger.warning("Using default discomfort scores based on token length")

    # Start iterative token selection
    logger.info("Starting iterative token selection")
    benchmark.start_phase(BenchmarkPhase.SET_IMPROVEMENT)

    # Choose whether to use parallel processing
    use_parallel = config.use_parallel_processing and len(corpus_text) > 100000

    final_tokens = select_tokens_iteratively(
        corpus_text,
        config.token_analysis.min_token_length,
        config.token_analysis.max_token_length,
        config.token_analysis.learning_limit_type,
        config.general.chords_to_assign,
        config.token_analysis.top_n_tokens,
        word_set_id,
        benchmark,
        debug_options,
        layout_comfort=layout_comfort,
        use_parallel=use_parallel,
    )

    benchmark.end_phase()

    # Create output filename
    corpus_name = os.path.splitext(config.active_corpus_file)[0]
    output_filename = f"{corpus_name}_tokens_{config.token_analysis.top_n_tokens}.json"
    output_path = config.paths.tokens_dir / output_filename

    # Save intermediate results if enabled
    if debug_options.get("save_intermediate", False):
        # Save a debug version with discomfort scores visible in filename
        debug_output_filename = (
            f"{corpus_name}_tokens_{config.token_analysis.top_n_tokens}_debug.json"
        )
        debug_output_path = config.paths.debug_dir / debug_output_filename
        logger.info(f"Saving debug token information to {debug_output_path}")
        final_tokens.save_to_file(debug_output_path)

    # Save to file
    logger.info(f"Saving {len(final_tokens.tokens)} tokens to {output_path}")
    benchmark.start_phase(BenchmarkPhase.WRITING_OUTPUT)
    final_tokens.save_to_file(output_path)
    benchmark.end_phase()

    # Log benchmark results if enabled
    if config.benchmark.enabled:
        results = benchmark.get_results()
        logger.info(f"Benchmark results: {results}")

    # Log selected token summary
    selected_tokens = [t for t in final_tokens.tokens if t.selected]
    logger.info(
        f"Selected {len(selected_tokens)} tokens out of {len(final_tokens.tokens)} total tokens"
    )

    # Log token type distribution
    token_type_counts = {t.name: 0 for t in TokenType}
    for t in selected_tokens:
        token_type_counts[t.token_type.name] += 1

    logger.info(f"Selected token type distribution: {token_type_counts}")

    # Log top 10 selected tokens
    top_selected = (
        selected_tokens[:10] if len(selected_tokens) >= 10 else selected_tokens
    )
    token_info = "\n".join(
        f"  {i+1}. '{t.lower}' (discomfort: {t.score:.4f}, count: {t.text_count}, type: {t.token_type.name})"
        for i, t in enumerate(top_selected)
    )
    logger.info(f"Top selected tokens:\n{token_info}")

    logger.info("Token extraction completed successfully")
