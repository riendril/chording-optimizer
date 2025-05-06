"""
Token extraction module for chord optimization.

This module handles extracting tokens of various types from text,
providing a clean API for token generation.
"""

import logging
import re
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, List

from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """Clean and normalize text for token extraction."""
    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_character_tokens(text: str) -> CounterType:
    """Extract individual characters and their frequencies."""
    return Counter(text)


def extract_character_ngrams(text: str, n: int) -> CounterType:
    """Extract character n-grams and their frequencies."""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def extract_word_tokens(text: str) -> CounterType:
    """Extract word tokens and their frequencies."""
    return Counter(re.findall(r"\b[\w\']+\b", text))


def extract_word_tokens_with_space(text: str) -> CounterType:
    """Extract word tokens with trailing space and their frequencies."""
    return Counter(re.findall(r"\b[\w\']+\s", text))


def extract_word_ngrams(text: str, n: int) -> CounterType:
    """Extract word n-grams and their frequencies."""
    words = re.findall(r"\b[\w\']+\b", text)
    return Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def extract_punctuation_patterns(text: str) -> CounterType:
    """Extract common punctuation patterns and their frequencies."""
    patterns = [
        r"\.{2,}",  # multiple periods
        r"[,\.;:][\'\"]",  # punctuation followed by quotes
        r"\w+\.\w+",  # domain-like patterns
        r"[!?]{2,}",  # multiple ! or ?
    ]

    all_matches = []
    for pattern in patterns:
        all_matches.extend(re.findall(pattern, text))

    return Counter(all_matches)


def extract_tokens_from_text(text: str, config: GeneratorConfig) -> Dict[str, int]:
    """Extract tokens from text based on configuration."""
    logger.info("Preprocessing text...")
    processed_text = preprocess_text(text)

    logger.info("Extracting tokens...")
    token_config = config.token_analysis
    counters = []

    # Add character tokens if enabled
    if token_config.include_characters:
        counters.append(extract_character_tokens(processed_text))

    # Add character n-grams if enabled
    if token_config.include_character_ngrams:
        min_n = max(2, token_config.min_token_length)
        max_n = min(5, token_config.max_token_length)  # Cap at 5 for character n-grams
        for n in range(min_n, max_n + 1):
            counters.append(extract_character_ngrams(processed_text, n))

    # Add tokens if enabled
    if token_config.include_words:
        counters.append(extract_word_tokens(processed_text))
        counters.append(extract_word_tokens_with_space(processed_text))

    # Add token n-grams if enabled
    if token_config.include_token_ngrams:
        min_n = max(2, token_config.min_token_length)
        max_n = min(4, token_config.max_token_length)  # Cap at 4 for token n-grams
        for n in range(min_n, max_n + 1):
            counters.append(extract_word_ngrams(processed_text, n))

    # Add punctuation patterns
    counters.append(extract_punctuation_patterns(processed_text))

    # Merge all counters
    merged_counter = reduce(lambda x, y: x + y, counters, Counter())

    logger.info(f"Found {len(merged_counter)} unique tokens")
    return merged_counter


def create_token_collection(
    tokens: Dict[str, int], name: str, source: str
) -> TokenCollection:
    """Create a TokenCollection from token frequencies."""
    # Sort tokens by frequency
    sorted_tokens = sorted(
        [(token, freq) for token, freq in tokens.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # Create TokenData objects
    token_objects = []
    for i, (token, freq) in enumerate(sorted_tokens):
        # Calculate Zipf weight (simple approximation)
        zipf_weight = 1.0 / (i + 1)

        # Create TokenData
        token_data = TokenData.from_token(
            token=token,
            frequency=freq,
            rank=i,
            zipf_weight=zipf_weight,
            score=freq * zipf_weight,  # Simple initial score
        )

        token_objects.append(token_data)

    # Create and return the collection
    return TokenCollection(
        name=name, tokens=token_objects, ordered_by_frequency=True, source=source
    )


def extract_tokens(config: GeneratorConfig) -> None:
    """
    Main entry point: Extract tokens from corpus file and save to JSON.

    This function reads the corpus specified in the config, extracts tokens
    according to the config settings, and saves the top N tokens to a JSON file.

    Args:
        config: Generator configuration with all necessary parameters
    """
    # Get paths from config
    corpus_path = config.paths.get_corpus_path(config.active_corpus_file)
    top_n = config.token_analysis.top_n_tokens

    # Construct output filename
    output_path = config.paths.tokens_dir / f"{corpus_path.stem}_tokens_{top_n}.json"

    logger.info(f"Extracting tokens from {corpus_path}")

    # Read corpus file
    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus_text = file.read()

    # Extract all tokens
    all_tokens = extract_tokens_from_text(corpus_text, config)

    # Limit to top N tokens
    if top_n < len(all_tokens):
        logger.info(f"Limiting to top {top_n} tokens")
        sorted_tokens = sorted(
            [(t, f) for t, f in all_tokens.items()], key=lambda x: x[1], reverse=True
        )[:top_n]
        all_tokens = {t: f for t, f in sorted_tokens}

    # Create token collection
    collection_name = f"{corpus_path.stem}_tokens_{top_n}"
    collection = create_token_collection(
        all_tokens, collection_name, source=str(corpus_path)
    )

    # Save to file
    collection.save_to_file(output_path)

    logger.info(f"Saved {len(collection.tokens)} tokens to {output_path}")
