"""
Token extraction and classification module.

This module provides functionality to extract words from text and classify tokens.
"""

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from src.common.shared_types import TokenData, TokenType

logger = logging.getLogger(__name__)

# Global variable to hold the word set for caching
_word_set_for_cache = set()


def extract_words_from_text(text: str) -> set[str]:
    """Extract real words from text using word boundaries.

    Args:
        text: Text to extract words from

    Returns:
        set of lowercase words found in the text
    """
    # Regular expression to find words (sequences of letters)
    # \b represents word boundary
    word_pattern = r"\b[a-zA-Z]+\b"
    words = re.findall(word_pattern, text)

    # Convert to lowercase and store in a set for efficient lookup
    return {word.lower() for word in words}


@lru_cache(maxsize=100000)  # Cache classification results for up to 100,000 tokens
def classify_token(token: str, word_set_id: int) -> TokenType:
    """Classify a token into its type category using word context.

    This function is cached to improve performance for repeated tokens.

    Args:
        token: The token string to classify
        word_set_id: A unique identifier for the word set (used for cache key)

    Returns:
        TokenType enumeration value
    """
    # We need to access the word_set via the global reference
    word_set = _word_set_for_cache

    # Single character
    if len(token) == 1:
        return TokenType.SINGLE_CHARACTER

    # Check if token is in our set of known words
    token_lower = token.lower()
    token_without_space = token_lower.rstrip()

    # Full word check (using our word set for validation)
    if token_lower in word_set:
        return TokenType.FULL_WORD

    # Word followed by space
    if token_lower.endswith(" ") and token_without_space in word_set:
        return TokenType.WORD_FOLLOWED_BY_SPACE

    # Check if it consists only of letters (but not a known word)
    if token.isalpha():
        return TokenType.NGRAM_LETTERS_ONLY

    # Check if it's an n-gram with no letters
    if not any(c.isalpha() for c in token):
        return TokenType.NGRAM_NO_LETTERS

    # Default case
    return TokenType.OTHER


def set_word_set_for_cache(word_set: set[str]) -> int:
    """Set the global word set for cache and return a unique identifier.

    Args:
        word_set: The set of words to use for classification

    Returns:
        A unique identifier for this word set (used for cache key)
    """
    global _word_set_for_cache
    _word_set_for_cache = word_set
    # Using id of the set as a unique identifier
    return id(word_set)


def extract_tokens_from_segmentation(
    segmentation: List[Tuple[str, int, int, TokenData]],
    text: str,
    min_token_length: int,
    max_token_length: int,
    word_set_id: int,
) -> List[TokenData]:
    """Extract all token candidates from current optimal text segmentation.

    Args:
        segmentation: List of (token_text, start_pos, end_pos, token_data)
                     from optimal segmentation
        text: Original text
        min_token_length: Minimum subtoken count in a token
        max_token_length: Maximum subtoken count in a token
        word_set_id: Identifier for word set used in classification

    Returns:
        List of extracted TokenData objects
    """
    # TODO: Why does this function even need the text? The segmentation should
    # be enough, right?

    # Counter for token frequencies
    from collections import Counter

    token_counter = Counter()
    token_compositions = {}

    # Sliding window on segmented positions
    n = len(segmentation)

    for window_size in range(min_token_length, max_token_length + 1):
        for i in range(n - window_size + 1):
            # Extract window of tokens
            window = segmentation[i : i + window_size]

            # Get text span
            start_pos = window[0][1]
            end_pos = window[-1][2]
            token_text = text[start_pos:end_pos].lower()

            # Count this token
            token_counter[token_text] += 1

            # Store the composition (which selected tokens make up this new token)
            composition = [w[0] for w in window]
            # FIX: Use the TokenData field instead of randomly implementing a
            # separate data type (best_current_combination)

            if token_text not in token_compositions:
                token_compositions[token_text] = composition

            # If this is a more optimal composition, update it
            elif len(composition) < len(token_compositions[token_text]):
                token_compositions[token_text] = composition

    # Create TokenData objects from counter
    token_data_list = []

    for token_text, count in token_counter.items():
        # Classify token
        token_type = classify_token(token_text, word_set_id)

        # Create TokenData
        token_data = TokenData(
            lower=token_text,
            length=len(token_text),
            token_type=token_type,
            text_count=count,
            usage_count=count,
            rank=0,  # Will be assigned later
            score=0.0,  # Will be calculated later
            selected=False,
            best_current_combination=token_compositions.get(token_text, [token_text]),
        )

        token_data_list.append(token_data)

    return token_data_list


def extract_tokens_from_segmentation_parallel(
    segmentation: List[Tuple[str, int, int, TokenData]],
    text: str,
    min_token_length: int,
    max_token_length: int,
    word_set_id: int,
) -> List[TokenData]:
    """Extract new token candidates from optimal text segmentation using parallel processing.

    Args:
        segmentation: List of (token_text, start_pos, end_pos, token_data)
        text: Original text
        min_token_length: Minimum token count
        max_token_length: Maximum token count
        word_set_id: Identifier for word set

    Returns:
        List of extracted TokenData objects
    """
    # TODO: Why does this function even need the text? The segmentation should
    # be enough, right?
    # TODO: Why is this even such a complicated function in the first place? All
    # one needs to do after the segmentation is simply count all unique
    # combinations of segments between min_length and max_length

    import multiprocessing

    # For small segmentations, use sequential approach
    if len(segmentation) < 1000:
        return extract_tokens_from_segmentation(
            segmentation, text, min_token_length, max_token_length, word_set_id
        )

    # Create a manager for sharing data between processes
    # This is needed for the global word set cache
    manager = multiprocessing.Manager()
    shared_word_set = manager.list(_word_set_for_cache)

    # Split work into chunks for parallel processing
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(100, len(segmentation) // cpu_count)

    # Create work chunks with overlap to ensure we don't miss tokens at boundaries
    chunks = []
    n = len(segmentation)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size + max_token_length - 1, n)
        chunks.append(
            (segmentation[i:end], text, min_token_length, max_token_length, word_set_id)
        )

    # Process chunks in parallel
    with multiprocessing.Pool() as pool:
        results = pool.starmap(extract_tokens_from_segmentation, chunks)

    # Combine results
    all_tokens = {}

    for chunk_tokens in results:
        for token in chunk_tokens:
            key = token.lower
            if key in all_tokens:
                # Merge frequencies
                all_tokens[key].text_count += token.text_count
                all_tokens[key].usage_count += token.usage_count

                # Keep the best composition (shorter is better)
                if len(token.best_current_combination) < len(
                    all_tokens[key].best_current_combination
                ):
                    all_tokens[key].best_current_combination = (
                        token.best_current_combination
                    )
            else:
                all_tokens[key] = token

    return list(all_tokens.values())
