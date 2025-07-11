"""
Token selection module with iterative selection algorithm.

This module processes corpus text to:
1. Initialize a list of selected tokens (starting with single characters)
2. Iteratively find optimal segmentation of text using currently selected tokens
3. Generate new token candidates from the optimally segmented text
4. Score all current tokens (usage cost and replacement score)
5. Select next eligible token based on replacement score
6. Output a ranked list of tokens
"""

import logging
import os
from typing import Dict, Optional

import tqdm

from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData, TokenType
from src.token_generation.text_segmentation import (
    find_optimal_text_segmentation,
    visualize_text_segmentation,
)
from src.token_generation.token_context import add_adjacency_context_to_tokens
from src.token_generation.token_extraction import (
    extract_tokens_from_segmentation,
    extract_tokens_from_segmentation_parallel,
    extract_words_from_text,
    set_word_set_for_cache,
)
from src.token_generation.token_scoring import update_token_scores_and_sort

logger = logging.getLogger(__name__)


def select_tokens_iteratively(
    text: str,
    min_token_length: int,
    max_token_length: int,
    learning_limit: TokenType,
    chords_to_assign: int,
    top_n_tokens: int,
    benchmark: Benchmark,
    debug_options: dict,
    layout_usage_cost: Dict[str, float],
    use_parallel: bool,
    pre_selected_tokens: Optional[TokenCollection],
) -> TokenCollection:
    """Select tokens iteratively by finding optimal segmentation after each selection.

    Args:
        text: Input corpus text
        min_token_length: Minimum token length
        max_token_length: Maximum token length
        learning_limit: Maximum token type complexity
        chords_to_assign: Number of tokens to select
        top_n_tokens: Number of top tokens to include in output
        benchmark: Benchmark instance
        debug_options: Debugging options
        layout_usage_cost: Dict mapping characters to usage costs
        use_parallel: Whether to use parallel processing

    Returns:
        TokenCollection with selected tokens and top candidates
    """
    # Initialize with single letter tokens
    selected_tokens = []

    # Text length for normalization
    text_length = len(text)

    # Create all single letter tokens
    unique_chars = set(text.lower())
    logger.info(f"Found {len(unique_chars)} unique characters in corpus")

    for char in unique_chars:
        token_data = TokenData(
            lower=char,
            character_length=1,
            subtoken_length=1,
            token_type=TokenType.SINGLE_CHARACTER,
            text_count=text.lower().count(char),
            usage_count=text.lower().count(char),
            rank=0,
            usage_cost=0.0,
            replacement_score=0.0,
            selected=True,
            best_current_combination=[char],
            adjacent_tokens=None,
        )

        selected_tokens.append(token_data)

    # Add pre-configured tokens if provided
    if pre_selected_tokens:
        logger.info(f"Loading {len(pre_selected_tokens.tokens)} pre-configured tokens")
        for token in pre_selected_tokens.tokens:
            # Check if this token already exists (avoid duplicates)
            if not any(existing.lower == token.lower for existing in selected_tokens):
                # Create a copy and mark as selected
                pre_token = TokenData(
                    lower=token.lower,
                    character_length=token.character_length,
                    subtoken_length=1,  # Will be updated later
                    token_type=token.token_type,
                    text_count=text.lower().count(token.lower),
                    usage_count=text.lower().count(token.lower),
                    rank=0,
                    usage_cost=1.0,  # Will be updated later
                    replacement_score=0.0,
                    selected=True,
                    best_current_combination=[token.lower],
                )
                selected_tokens.append(pre_token)
            else:
                logger.debug(
                    f"Pre-configured token '{token.lower}' already exists as single character"
                )

    # Calculate scores for all selected tokens
    update_token_scores_and_sort(
        selected_tokens, text_length, selected_tokens, layout_usage_cost
    )

    contained_character_count = len(
        [t for t in selected_tokens if t.token_type == TokenType.SINGLE_CHARACTER]
    )
    pre_configured_count = len(selected_tokens) - contained_character_count
    logger.info(
        f"Initialized with {contained_character_count} single character tokens and {pre_configured_count} pre-configured tokens"
    )

    # Main selection loop
    progress_bar = tqdm.tqdm(
        total=chords_to_assign,
        desc="Selecting tokens iteratively",
    )

    # Keep track of iteration for debugging
    iteration = 0
    current_token_candidates = []
    current_segmentation = []

    while True:
        iteration += 1

        # Find optimal segmentation using currently selected tokens
        current_segmentation = find_optimal_text_segmentation(text, selected_tokens)

        # Debug: Visualize segmentation if enabled
        if debug_options.get("print_segmentation", False):
            visualization = visualize_text_segmentation(current_segmentation, 100, 1)
            logger.info(
                f"Segmentation visualization (iteration {iteration}):\n{visualization}"
            )

        # Extract current token candidates
        if use_parallel:
            current_token_candidates = extract_tokens_from_segmentation_parallel(
                current_segmentation, min_token_length, max_token_length
            )
        else:
            current_token_candidates = extract_tokens_from_segmentation(
                current_segmentation, min_token_length, max_token_length
            )

        # Update scores of selected_tokens
        update_token_scores_and_sort(
            selected_tokens, text_length, selected_tokens, layout_usage_cost
        )

        # Calculate scores for all candidates
        update_token_scores_and_sort(
            current_token_candidates, text_length, selected_tokens, layout_usage_cost
        )

        # Debug: show top candidates if enabled
        if debug_options.get("print_candidates", False):
            top_candidates = (
                current_token_candidates[:10]
                if len(current_token_candidates) > 10
                else current_token_candidates
            )
            candidate_info = "\n".join(
                f"  {i+1}. '{c.lower}' "
                f"(replacement_score: {c.replacement_score:.6f}, "
                f"usage_cost: {c.usage_cost:.4f}, "
                f"usage_count: {c.usage_count}, type: {c.token_type.name})"
                for i, c in enumerate(top_candidates)
            )
            logger.info(f"Top candidates (iteration {iteration}):\n{candidate_info}")

        if (
            len(selected_tokens) - contained_character_count - pre_configured_count
            >= chords_to_assign
        ):
            break

        next_token = None

        # Find the highest scoring eligible candidate
        for candidate in current_token_candidates:
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

        # Add to selected tokens and update values
        next_token.selected = True
        next_token.subtoken_length = 1
        # TODO: LATER assign chord and chords usage_cost instead
        next_token.best_current_combination = [next_token.lower]
        next_token.usage_cost = 2
        selected_tokens.append(next_token)

        # Log with both usage cost and replacement score
        logger.info(
            f"Selected token: '{next_token.lower}' "
            f"(replacement_score: {next_token.replacement_score:.6f}, "
            f"usage_cost: {next_token.usage_cost:.4f}, "
            f"usage_count: {next_token.usage_count}, type: {next_token.token_type.name})"
        )

        # Update progress
        progress_bar.update(1)
        benchmark.update_phase(len(selected_tokens))

    progress_bar.close()

    final_segmentation = current_segmentation

    # Add adjacency context to selected tokens
    logger.info("Adding adjacency context to selected tokens...")
    add_adjacency_context_to_tokens(selected_tokens, final_segmentation)

    # Find the lowest scoring selected token
    lowest_selected_score = min(t.replacement_score for t in selected_tokens)

    # Find unselected tokens that score higher than the lowest selected
    recommendations = [
        t
        for t in current_token_candidates
        if not t.selected and t.replacement_score > lowest_selected_score
    ]

    logger.info(
        f"Found {len(recommendations)} unselected tokens scoring higher than lowest selected token (score: {lowest_selected_score:.6f})"
    )

    # Log top 5 recommendations
    top_recommendations = recommendations[:5]
    rec_info = "\n".join(
        f"  {i+1}. '{r.lower}' (score: {r.replacement_score:.6f}, type: {r.token_type.name})"
        for i, r in enumerate(top_recommendations)
    )
    logger.info(f"Top recommendations for additional learning:\n{rec_info}")

    # Combine all tokens into one list, avoiding duplicates
    all_tokens = {}

    # Add selected tokens first
    for token in selected_tokens:
        all_tokens[token.lower] = token

    # Add candidates, but only if they're not already in selected_tokens
    # TODO:Tokens being in both lists should not be possible in the first place!
    # Check code for this, then put an assertion here!!
    for token in current_token_candidates:
        if token.lower not in all_tokens:
            all_tokens[token.lower] = token

    # Convert back to list and sort by replacement score
    final_tokens_list = list(all_tokens.values())

    # Assign ranks based on sorted order
    for i, token in enumerate(final_tokens_list):
        token.rank = i + 1

    # Take only the top_n_tokens
    final_tokens_list = final_tokens_list[:top_n_tokens]

    # Visualize final segmentation if enabled
    if debug_options.get("print_segmentation"):
        visualization = visualize_text_segmentation(final_segmentation, 20, 3)
        logger.info(f"Final segmentation visualization:\n{visualization}")
    else:
        # Always show at least one visualization sample of the final segmentation
        visualization = visualize_text_segmentation(final_segmentation, 20, 1)
        logger.info(f"Sample of final segmentation:\n{visualization}")

    # Create token collection with final tokens
    token_collection = TokenCollection(
        name="iterative_selection",
        tokens=final_tokens_list,
        ordered_by_frequency=True,
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

    # Set global word set for caching
    set_word_set_for_cache(word_set)

    # Set up debug options based on config
    debug_options = {
        "print_segmentation": config.debug.enabled and config.debug.print_cost_details,
        "print_candidates": config.debug.enabled and config.debug.print_cost_details,
        "save_intermediate": config.debug.enabled
        and config.debug.save_intermediate_results,
    }

    # Load keyboard layout usage cost information (required)
    from src.common.layout import load_keyboard_layout

    layout_path = config.paths.get_layout_path(config.active_layout_file)
    layout_data = load_keyboard_layout(layout_path)

    if "usage_cost" not in layout_data:
        raise ValueError(
            f"Layout file '{config.active_layout_file}' missing required usage_cost matrix"
        )

    layout_usage_cost = layout_data["usage_cost"]
    logger.info(
        f"Loaded usage cost information from keyboard layout '{config.active_layout_file}'"
    )

    # Validate that all usage cost values are positive
    zero_values = [key for key, value in layout_usage_cost.items() if value <= 0]
    if zero_values:
        raise ValueError(
            f"Layout usage cost matrix contains non-positive values for keys: {zero_values}"
        )

    # Ensure 'unknown' key exists in layout usage cost
    if "unknown" not in layout_usage_cost:
        raise ValueError("Layout usage cost matrix missing required 'unknown' entry")

    # Load pre-configured tokens if specified
    pre_selected_tokens = None
    if config.preselected_tokens_file:
        try:
            tokens_path = config.paths.get_tokens_path(config.preselected_tokens_file)
            pre_selected_tokens = TokenCollection.load_from_file(tokens_path)
            logger.info(
                f"Loaded {len(pre_selected_tokens.tokens)} pre-configured tokens from {config.preselected_tokens_file}"
            )
        except Exception as e:
            logger.warning(
                f"Could not load pre-configured tokens from {config.preselected_tokens_file}: {e}"
            )

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
        benchmark,
        debug_options,
        layout_usage_cost=layout_usage_cost,
        use_parallel=use_parallel,
        pre_selected_tokens=pre_selected_tokens,
    )

    benchmark.end_phase()

    # Create output filename
    corpus_name = os.path.splitext(config.active_corpus_file)[0]
    output_filename = f"{corpus_name}_tokens_{config.token_analysis.top_n_tokens}.json"
    output_path = config.paths.tokens_dir / output_filename

    # Save intermediate results if enabled
    # TODO: these results are not intermediate, saving should be done during the
    # iterative selection
    if debug_options.get("save_intermediate"):
        # Save a debug version with usage cost scores visible in filename
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
    token_debug = "\n".join(
        f"  {i+1}. '{t.lower}' (replacement_score: {t.replacement_score:.4f}, usage_cost: {t.usage_cost:.4f}, usage_count: {t.usage_count}, type: {t.token_type.name})"
        for i, t in enumerate(top_selected)
    )
    logger.debug(f"Top selected tokens:\n{token_debug}")
    logger.debug("Token extraction completed successfully")
