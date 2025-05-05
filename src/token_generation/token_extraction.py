"""
Token extraction module for chord optimization.

This module handles extracting tokens of various types from text,
providing a clean API for token generation.
"""

import re
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for token extraction.

    Args:
        text: Raw input text

    Returns:
        Cleaned and normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------
# Token Extraction Functions
# -----------------


def extract_character_tokens(text: str) -> Dict[str, int]:
    """
    Extract individual characters and their frequencies.

    Args:
        text: Preprocessed text

    Returns:
        Dictionary mapping characters to their frequencies
    """
    return Counter(text)


def extract_character_ngrams(text: str, n: int) -> Dict[str, int]:
    """
    Extract character n-grams and their frequencies.

    Args:
        text: Preprocessed text
        n: Length of character sequences to extract

    Returns:
        Dictionary mapping character n-grams to their frequencies
    """
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def extract_word_tokens(text: str) -> Dict[str, int]:
    """
    Extract word tokens and their frequencies.

    Args:
        text: Preprocessed text

    Returns:
        Dictionary mapping words to their frequencies
    """
    return Counter(re.findall(r"\b[\w\']+\b", text))


def extract_word_tokens_with_space(text: str) -> Dict[str, int]:
    """
    Extract word tokens with trailing space and their frequencies.

    Args:
        text: Preprocessed text

    Returns:
        Dictionary mapping words with trailing space to their frequencies
    """
    return Counter(re.findall(r"\b[\w\']+\s", text))


def extract_word_ngrams(text: str, n: int) -> Dict[str, int]:
    """
    Extract word n-grams and their frequencies.

    Args:
        text: Preprocessed text
        n: Number of consecutive words in each n-gram

    Returns:
        Dictionary mapping word n-grams to their frequencies
    """
    words = re.findall(r"\b[\w\']+\b", text)
    return Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def extract_punctuation_patterns(text: str) -> Dict[str, int]:
    """
    Extract common punctuation patterns and their frequencies.

    Args:
        text: Preprocessed text

    Returns:
        Dictionary mapping punctuation patterns to their frequencies
    """
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


def merge_token_frequencies(counters: List[Counter]) -> Counter:
    """
    Merge multiple frequency counters into one.

    Args:
        counters: List of Counter objects to merge

    Returns:
        Merged Counter object
    """
    return reduce(lambda x, y: x + y, counters, Counter())


# -----------------
# Main Extraction Functions
# -----------------


def extract_tokens(text: str, config: GeneratorConfig) -> Dict[str, int]:
    """
    Extract all specified token types from text based on configuration.

    Args:
        text: Preprocessed text
        config: Generator configuration with token analysis settings

    Returns:
        Dictionary mapping tokens to their frequencies
    """
    token_config = config.token_analysis
    counters = []

    # Add character tokens if enabled
    if token_config.include_characters:
        counters.append(extract_character_tokens(text))

    # Add character n-grams if enabled
    if token_config.include_character_ngrams:
        min_n = max(2, token_config.min_token_length)
        max_n = min(5, token_config.max_token_length)  # Cap at 5 for character n-grams
        for n in range(min_n, max_n + 1):
            counters.append(extract_character_ngrams(text, n))

    # Add tokens if enabled
    if token_config.include_tokens:
        counters.append(extract_word_tokens(text))
        counters.append(extract_word_tokens_with_space(text))

    # Add token n-grams if enabled
    if token_config.include_token_ngrams:
        min_n = max(2, token_config.min_token_length)
        max_n = min(4, token_config.max_token_length)  # Cap at 4 for token n-grams
        for n in range(min_n, max_n + 1):
            counters.append(extract_word_ngrams(text, n))

    # Add punctuation patterns
    counters.append(extract_punctuation_patterns(text))

    # Merge all counters
    return merge_token_frequencies(counters)


# -----------------
# Public API
# -----------------


def extract_tokens_from_text(
    corpus: str, config: GeneratorConfig, show_progress: bool = True
) -> Dict[str, int]:
    """
    Extract tokens from text without building context information.

    Args:
        corpus: Raw input text
        config: Generator configuration
        show_progress: Whether to show progress updates

    Returns:
        Dictionary mapping tokens to their frequencies
    """
    if show_progress:
        print("Preprocessing text...")

    processed_text = preprocess_text(corpus)

    if show_progress:
        print("Extracting tokens...")

    tokens = extract_tokens(processed_text, config)

    if show_progress:
        print(f"Found {len(tokens)} unique tokens.")

    return tokens


def create_token_collection(
    tokens: Dict[str, int],
    name: str,
    source: Optional[str] = None,
    zipf_weight_base: float = 1.0,
) -> TokenCollection:
    """
    Create a TokenCollection without context information.

    Args:
        tokens: Dictionary mapping tokens to their frequencies
        name: Name for the collection
        source: Source identifier for the collection
        zipf_weight_base: Base value for Zipf weighting

    Returns:
        TokenCollection object
    """
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
        zipf_weight = zipf_weight_base / (i + 1)

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


# -----------------
# File I/O Functions
# -----------------


def read_corpus_from_file(file_path: Union[str, Path]) -> str:
    """
    Read corpus data from a file.

    Args:
        file_path: Path to corpus file

    Returns:
        File contents as string
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_tokens_from_file(
    file_path: Union[str, Path],
    config: GeneratorConfig,
    top_n: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    Extract tokens from a file without building context information.

    Args:
        file_path: Path to corpus file
        config: Generator configuration
        top_n: Number of top tokens to keep (by frequency)
        show_progress: Whether to show progress updates

    Returns:
        Dictionary mapping tokens to their frequencies
    """
    corpus = read_corpus_from_file(file_path)
    tokens = extract_tokens_from_text(corpus, config, show_progress)

    # Limit to top_n tokens if specified
    if top_n is not None and top_n < len(tokens):
        sorted_tokens = sorted(
            [(t, f) for t, f in tokens.items()], key=lambda x: x[1], reverse=True
        )[:top_n]

        tokens = {t: f for t, f in sorted_tokens}

    return tokens


def create_and_save_token_collection(
    corpus_path: Union[str, Path],
    output_path: Union[str, Path],
    config: GeneratorConfig,
    top_n: Optional[int] = None,
    collection_name: Optional[str] = None,
    show_progress: bool = True,
) -> TokenCollection:
    """
    Extract tokens from a file, create a collection, and save it.

    Args:
        corpus_path: Path to corpus file
        output_path: Path to save token collection
        config: Generator configuration
        top_n: Number of top tokens to keep (by frequency)
        collection_name: Name for the collection (defaults to corpus filename)
        show_progress: Whether to show progress updates

    Returns:
        The created TokenCollection
    """
    corpus_path = Path(corpus_path)
    if not collection_name:
        collection_name = f"{corpus_path.stem}_tokens_{top_n or 'all'}"

    tokens = extract_tokens_from_file(corpus_path, config, top_n, show_progress)
    collection = create_token_collection(
        tokens, collection_name, source=str(corpus_path)
    )

    # Save to file
    collection.save_to_file(output_path)

    if show_progress:
        print(f"Saved {len(collection.tokens)} tokens to {output_path}")

    return collection
