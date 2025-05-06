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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData

logger = logging.getLogger(__name__)


def extract_tokens(config: GeneratorConfig) -> None:
    """Extract, score, and order tokens from the corpus

    Args:
        config: Generator configuration
    """
    logger.info("Starting token extraction process")

    # Get corpus file path
    corpus_path = config.paths.get_corpus_path(config.active_corpus_file)

    # Load corpus text
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    # Extract all tokens within length constraints
    logger.info("Extracting tokens from corpus")
    token_counts = extract_all_tokens(
        corpus_text,
        config.token_analysis.min_token_length,
        config.token_analysis.max_token_length,
        config.use_parallel_processing,
    )

    # Score tokens based on frequency and length
    logger.info("Scoring tokens")
    token_scores = score_tokens(token_counts)

    # Get initial top tokens
    initial_top_tokens = get_top_n_tokens(
        token_scores, token_counts, config.token_analysis.top_n_tokens
    )

    # Apply subtoken/supertoken adjustments
    logger.info("Applying subtoken/supertoken adjustments")
    final_tokens = apply_token_adjustments(initial_top_tokens, token_counts)

    # Create output filename
    corpus_name = os.path.splitext(config.active_corpus_file)[0]
    output_filename = f"{corpus_name}_tokens_{config.token_analysis.top_n_tokens}.json"
    output_path = config.paths.tokens_dir / output_filename

    # Save to file
    logger.info(f"Saving {len(final_tokens.tokens)} tokens to {output_path}")
    final_tokens.save_to_file(output_path)

    logger.info("Token extraction completed successfully")


def extract_all_tokens(
    text: str, min_length: int, max_length: int, use_parallel: bool
) -> Dict[str, int]:
    """Extract all possible tokens within length constraints

    Args:
        text: Input corpus text
        min_length: Minimum token length to include
        max_length: Maximum token length to include
        use_parallel: Whether to use parallel processing

    Returns:
        Dictionary mapping tokens to their frequency counts
    """
    # Normalize text
    text = text.lower()

    # Process in parallel or sequentially
    if use_parallel and len(text) > 100000:  # Only parallelize for large corpora
        return _extract_tokens_parallel(text, min_length, max_length)

    token_counts = Counter()

    # Extract word tokens (words and word n-grams)
    words = re.findall(r"\b\w+\b", text)

    # Extract individual words
    for word in words:
        if min_length <= len(word) <= max_length:
            token_counts[word] += 1

    # Extract word n-grams
    for n in range(2, 4):  # Limit to 2-3 word phrases
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            if min_length <= len(ngram) <= max_length:
                token_counts[ngram] += 1

    # Extract character tokens (single characters and character n-grams)
    for n in range(min_length, max_length + 1):
        for i in range(len(text) - n + 1):
            ngram = text[i : i + n]
            # Skip n-grams with non-printable characters or excessive whitespace
            if all(c in string.printable for c in ngram) and not re.search(
                r"\s{2,}", ngram
            ):
                token_counts[ngram] += 1

    return token_counts


def _extract_tokens_parallel(
    text: str, min_length: int, max_length: int
) -> Dict[str, int]:
    """Extract tokens using parallel processing"""
    # Split text into chunks
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(1000, len(text) // cpu_count)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Create process pool
    with multiprocessing.Pool() as pool:
        # Process each chunk
        chunk_results = pool.starmap(
            extract_all_tokens,
            [(chunk, min_length, max_length, False) for chunk in chunks],
        )

    # Combine results
    combined_counts = Counter()
    for counts in chunk_results:
        combined_counts.update(counts)

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
    for token, count in token_counts.items():
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
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

    # Take top N tokens
    top_tokens = sorted_tokens[:top_n]

    # Create TokenData objects
    token_data_list = []
    for i, (token, score) in enumerate(top_tokens):
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


def apply_token_adjustments(
    token_collection: TokenCollection, token_counts: Dict[str, int]
) -> TokenCollection:
    """Apply subtoken/supertoken adjustments to the token collection

    For each token, adjust the scores of:
    - Subtokens: Reduce frequency by the frequency of the current token
    - Supertokens: Reduce effective length by (token length - 1)

    Args:
        token_collection: Initial collection of top tokens
        token_counts: Complete dictionary of token counts

    Returns:
        Adjusted token collection
    """
    # Make a copy of the original token list
    tokens = token_collection.tokens.copy()

    # Track processed tokens to avoid double-counting
    processed_tokens = set()

    # Process tokens in order (highest score first)
    for i, token in enumerate(tokens):
        if token.original in processed_tokens:
            continue

        processed_tokens.add(token.original)

        # Process any remaining tokens that might be affected
        for j in range(i + 1, len(tokens)):
            other_token = tokens[j]

            # Skip already processed tokens
            if other_token.original in processed_tokens:
                continue

            # Check if other_token is a subtoken of token
            if other_token.original in token.original:
                # Adjust frequency by subtracting the frequency of the current token
                adjusted_frequency = max(1, other_token.frequency - token.frequency)
                other_token.frequency = adjusted_frequency

                # Recalculate score
                other_token.score = (
                    adjusted_frequency / sum(token_counts.values())
                ) * len(other_token.original)

            # Check if other_token is a supertoken of token
            elif token.original in other_token.original:
                # Adjust effective length by subtracting (token length - 1)
                effective_length = max(
                    1, len(other_token.original) - (len(token.original) - 1)
                )

                # Recalculate score with adjusted length
                other_token.score = (
                    other_token.frequency / sum(token_counts.values())
                ) * effective_length

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


if __name__ == "__main__":
    # For standalone testing
    import argparse

    parser = argparse.ArgumentParser(description="Extract tokens from corpus file")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--corpus", type=str, help="Override corpus file")
    args = parser.parse_args()

    # Load configuration
    config = GeneratorConfig.load_config(args.config)

    # Override corpus file if specified
    if args.corpus:
        config.active_corpus_file = args.corpus

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run extraction
    extract_tokens(config)
