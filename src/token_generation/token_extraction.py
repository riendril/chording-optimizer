"""
Token extraction, counting, and scoring module.

This module processes corpus text to:
1. Extract all possible tokens within min/max length constraints
2. Score tokens based on frequency and length
3. Apply subtoken/supertoken adjustments
4. Output a ranked list of tokens for chord assignment
"""

import logging
import multiprocessing
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import tqdm

from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData

logger = logging.getLogger(__name__)


# Function defined at module level for multiprocessing compatibility
def process_chunk(chunk, min_length, max_length):
    """Process a single text chunk for token extraction.

    Args:
        chunk: Text chunk to process
        min_length: Minimum token length to include
        max_length: Maximum token length to include

    Returns:
        Counter of token frequencies
    """
    result = Counter()

    # Process words
    words = re.findall(r"\b\w+\b", chunk)
    for word in words:
        if min_length <= len(word) <= max_length:
            result[word] += 1

    # Process word n-grams
    for n in range(2, 4):  # Limit to 2-3 word phrases
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            if min_length <= len(ngram) <= max_length:
                result[ngram] += 1

    # Process character n-grams
    for n in range(min_length, max_length + 1):
        for i in range(len(chunk) - n + 1):
            ngram = chunk[i : i + n]
            if all(c in string.printable for c in ngram) and not re.search(
                r"\s{2,}", ngram
            ):
                result[ngram] += 1

    return result


def extract_tokens(config: GeneratorConfig) -> None:
    """Extract, score, and order tokens from the corpus

    Args:
        config: Generator configuration
    """
    # Initialize benchmarking if enabled
    benchmark = Benchmark(config.benchmark)

    logger.info("Starting token extraction process")

    # Get corpus file path
    corpus_path = config.paths.get_corpus_path(config.active_corpus_file)

    # Load corpus text
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    # Extract all tokens within length constraints
    logger.info("Extracting tokens from corpus")
    benchmark.start_phase(BenchmarkPhase.INITIALIZATION)
    token_counts = extract_all_tokens(
        corpus_text,
        config.token_analysis.min_token_length,
        config.token_analysis.max_token_length,
        config.use_parallel_processing,
        benchmark,
    )
    benchmark.end_phase()

    # Score tokens based on frequency and length
    logger.info("Scoring tokens")
    benchmark.start_phase(BenchmarkPhase.CHORD_COST_CALCULATION)
    token_scores = score_tokens(token_counts)
    benchmark.end_phase()

    # Get initial top tokens
    logger.info(f"Selecting top {config.token_analysis.top_n_tokens} tokens")
    initial_top_tokens = get_top_n_tokens(
        token_scores, token_counts, config.token_analysis.top_n_tokens
    )

    # Apply subtoken/supertoken adjustments
    logger.info("Applying subtoken/supertoken adjustments")
    benchmark.start_phase(BenchmarkPhase.SET_IMPROVEMENT)
    final_tokens = apply_token_adjustments(initial_top_tokens, token_counts, benchmark)
    benchmark.end_phase()

    # Create output filename
    corpus_name = os.path.splitext(config.active_corpus_file)[0]
    output_filename = f"{corpus_name}_tokens_{config.token_analysis.top_n_tokens}.json"
    output_path = config.paths.tokens_dir / output_filename

    # Save to file
    logger.info(f"Saving {len(final_tokens.tokens)} tokens to {output_path}")
    benchmark.start_phase(BenchmarkPhase.WRITING_OUTPUT)
    final_tokens.save_to_file(output_path)
    benchmark.end_phase()

    # Log benchmark results if enabled
    if config.benchmark.enabled:
        results = benchmark.get_results()
        logger.info(f"Benchmark results: {results}")

    logger.info("Token extraction completed successfully")


def extract_all_tokens(
    text: str,
    min_length: int,
    max_length: int,
    use_parallel: bool,
    benchmark: Benchmark,
) -> Dict[str, int]:
    """Extract all possible tokens within length constraints

    Args:
        text: Input corpus text
        min_length: Minimum token length to include
        max_length: Maximum token length to include
        use_parallel: Whether to use parallel processing
        benchmark: Benchmark instance for performance tracking

    Returns:
        Dictionary mapping tokens to their frequency counts
    """
    # Normalize text
    text = text.lower()

    # Process in parallel or sequentially
    if use_parallel and len(text) > 100000:  # Only parallelize for large corpora
        return _extract_tokens_parallel(text, min_length, max_length, benchmark)

    token_counts = Counter()

    # Extract word tokens (words and word n-grams)
    words = re.findall(r"\b\w+\b", text)

    # Extract individual words with progress bar
    logger.info("Extracting words...")
    for word in tqdm.tqdm(words, desc="Extracting words"):
        if min_length <= len(word) <= max_length:
            token_counts[word] += 1
        benchmark.update_phase(len(token_counts))

    # Extract word n-grams with progress bar
    logger.info("Extracting word n-grams...")
    for n in range(2, 4):  # Limit to 2-3 word phrases
        for i in tqdm.tqdm(
            range(len(words) - n + 1), desc=f"Extracting {n}-word phrases"
        ):
            ngram = " ".join(words[i : i + n])
            if min_length <= len(ngram) <= max_length:
                token_counts[ngram] += 1
            benchmark.update_phase(len(token_counts))

    # Extract character tokens (single characters and character n-grams)
    logger.info("Extracting character n-grams...")

    # For large texts, we'll process in chunks for better progress tracking
    chunk_size = 10000
    total_chunks = (len(text) + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm.tqdm(range(total_chunks), desc="Processing text chunks"):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(text))
        chunk = text[chunk_start:chunk_end]

        for n in range(min_length, max_length + 1):
            for i in range(len(chunk) - n + 1):
                ngram = chunk[i : i + n]
                # Skip n-grams with non-printable characters or excessive whitespace
                if all(c in string.printable for c in ngram) and not re.search(
                    r"\s{2,}", ngram
                ):
                    token_counts[ngram] += 1

        benchmark.update_phase(len(token_counts))

    return token_counts


def _extract_tokens_parallel(
    text: str, min_length: int, max_length: int, benchmark: Benchmark
) -> Dict[str, int]:
    """Extract tokens using parallel processing"""
    # Split text into chunks
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(1000, len(text) // cpu_count)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    logger.info(f"Processing text in {len(chunks)} parallel chunks...")

    # Create process pool
    with multiprocessing.Pool() as pool:
        # Process each chunk with progress indication
        results = []

        for chunk in chunks:
            results.append(
                pool.apply_async(process_chunk, args=(chunk, min_length, max_length))
            )

        # Collect results with progress bar
        combined_counts = Counter()
        for result in tqdm.tqdm(results, desc="Processing chunks"):
            chunk_counts = result.get()
            combined_counts.update(chunk_counts)
            benchmark.update_phase(len(combined_counts))

    return combined_counts


def score_tokens(token_counts: Dict[str, int]) -> Dict[str, float]:
    """Score tokens based on frequency and length

    Args:
        token_counts: Dictionary of token frequencies

    Returns:
        Dictionary mapping tokens to their scores
    """
    # Find total token count for normalization
    total_count = sum(token_counts.values())

    # Calculate scores
    scores = {}
    for token, count in tqdm.tqdm(token_counts.items(), desc="Scoring tokens"):
        # Base score is normalized frequency
        frequency_score = count / total_count

        # Multiply by length for length benefit
        scores[token] = frequency_score * len(token)

    return scores


def get_top_n_tokens(
    token_scores: Dict[str, float], token_counts: Dict[str, int], top_n: int
) -> TokenCollection:
    """Get the top N tokens by score

    Args:
        token_scores: Dictionary mapping tokens to their scores
        token_counts: Dictionary mapping tokens to their frequencies
        top_n: Number of top tokens to include

    Returns:
        TokenCollection with the top tokens
    """
    # Sort tokens by score in descending order
    logger.info("Sorting tokens by score...")
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

    # Take top N tokens
    top_tokens = sorted_tokens[:top_n]

    # Create TokenData objects
    token_data_list = []
    for i, (token, score) in enumerate(
        tqdm.tqdm(top_tokens, desc="Creating token data")
    ):
        rank = i + 1

        token_data = TokenData.from_token(
            token=token, frequency=token_counts[token], rank=rank, score=score
        )
        token_data_list.append(token_data)

    # Create and return the collection
    return TokenCollection(
        name=f"top_{top_n}_tokens",
        tokens=token_data_list,
        ordered_by_frequency=True,
        source="corpus_extraction",
    )


def build_prefix_map(tokens: List[TokenData]) -> Dict[str, List[TokenData]]:
    """Build an efficient prefix map to find subtokens

    Args:
        tokens: List of TokenData objects

    Returns:
        Dictionary mapping tokens to all their possible subtokens
    """
    # Group tokens by their first character for faster lookups
    first_char_map = defaultdict(list)
    for token in tokens:
        if token.lower:  # Skip empty tokens
            first_char_map[token.lower[0]].append(token)

    # Build the prefix map
    prefix_map = {}
    for token in tokens:
        if not token.lower:  # Skip empty tokens
            continue

        # Find all possible subtokens
        subtokens = []
        token_text = token.lower

        # For each possible starting position in the token
        for start in range(len(token_text)):
            # Only check tokens that start with this character (optimization)
            possible_matches = first_char_map.get(token_text[start], [])

            # For each possible ending position
            for end in range(start + 1, len(token_text) + 1):
                substring = token_text[start:end]

                # Check if this substring is in our token list
                for potential_match in possible_matches:
                    if potential_match.lower == substring:
                        subtokens.append(potential_match)

        prefix_map[token.lower] = subtokens

    return prefix_map


def apply_token_adjustments(
    token_collection: TokenCollection,
    token_counts: Dict[str, int],
    benchmark: Benchmark,
) -> TokenCollection:
    """Apply subtoken/supertoken adjustments to the token collection

    For each token, adjust the scores of:
    - Subtokens: Reduce frequency by the frequency of the current token
    - Supertokens: Reduce effective length by (token length - 1)

    Args:
        token_collection: Initial collection of top tokens
        token_counts: Complete dictionary of token counts
        benchmark: Benchmark instance for performance tracking

    Returns:
        Adjusted token collection
    """
    # Make a copy of the original token list
    tokens = token_collection.tokens.copy()

    # Build efficient data structures for subtoken/supertoken lookup
    logger.info("Building token relationship maps...")

    # Build a map of token text to token object for quick lookup
    token_map = {token.lower: token for token in tokens}

    # Build prefix map for efficient subtoken detection
    prefix_map = build_prefix_map(tokens)

    # Build suffix map for efficient supertoken detection
    # We'll use a simpler approach: group tokens by length and then check
    tokens_by_length = defaultdict(list)
    for token in tokens:
        tokens_by_length[len(token.lower)].append(token)

    # Track processed tokens to avoid double-counting
    processed_tokens = set()
    total_tokens = len(tokens)

    # Apply adjustments with progress tracking
    logger.info("Applying token adjustments...")
    for i, token in enumerate(tqdm.tqdm(tokens, desc="Adjusting token scores")):
        if token.lower in processed_tokens:
            continue

        processed_tokens.add(token.lower)

        # Find subtokens (tokens contained within this token)
        for subtoken in prefix_map.get(token.lower, []):
            if subtoken.lower in processed_tokens or subtoken.lower == token.lower:
                continue

            # Adjust frequency by subtracting the frequency of the current token
            adjusted_frequency = max(1, subtoken.frequency - token.frequency)
            subtoken.frequency = adjusted_frequency

            # Recalculate score
            subtoken.score = (adjusted_frequency / sum(token_counts.values())) * len(
                subtoken.lower
            )

        # Find supertokens (tokens that contain this token)
        # More efficient approach: only check tokens longer than this one
        for length in range(
            len(token.lower) + 1,
            min(len(token.lower) * 2, max(tokens_by_length.keys()) + 1),
        ):
            for supertoken in tokens_by_length[length]:
                if (
                    supertoken.lower in processed_tokens
                    or supertoken.lower == token.lower
                ):
                    continue

                if token.lower in supertoken.lower:
                    # Adjust effective length by subtracting (token length - 1)
                    effective_length = max(
                        1, len(supertoken.lower) - (len(token.lower) - 1)
                    )

                    # Recalculate score with adjusted length
                    supertoken.score = (
                        supertoken.frequency / sum(token_counts.values())
                    ) * effective_length

        benchmark.update_phase(i)

    # Re-sort tokens by adjusted score
    tokens.sort(key=lambda t: t.score, reverse=True)

    # Update ranks
    for i, token in enumerate(tokens):
        token.rank = i + 1

    # Create new collection with adjusted tokens
    return TokenCollection(
        name=token_collection.name,
        tokens=tokens,
        ordered_by_frequency=True,
        source=token_collection.source,
    )
