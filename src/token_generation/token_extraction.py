"""
Token extraction and classification module.

This module provides functionality to extract words from text and classify tokens.
"""

import logging
import multiprocessing
import re
from collections import Counter
from functools import lru_cache
from typing import List

from src.common.shared_types import TokenData, TokenType
from src.token_generation.text_segmentation import TextSegment

logger = logging.getLogger(__name__)

# Global variable to hold the word set for caching
_word_set_for_cache = set()


def extract_words_from_text(text: str) -> set[str]:
    """Extract real words from text using word boundaries.

    Args:
        text: Text to extract words from (should already be lowercase)

    Returns:
        set of words found in the text
    """
    # Regular expression to find words (sequences of letters)
    # \b represents word boundary
    word_pattern = r"\b[a-zA-Z]+\b"
    words = re.findall(word_pattern, text)

    # Store in a set for efficient lookup
    return {word.lower() for word in words}


@lru_cache(maxsize=100000)  # Cache classification results for up to 100,000 tokens
def classify_token(token: str) -> TokenType:
    """Classify a token into its type category using word context.

    This function is cached to improve performance for repeated tokens.

    Args:
        token: The token string to classify (should already be lowercase)

    Returns:
        TokenType enumeration value
    """
    # Access the global word_set directly
    word_set = _word_set_for_cache

    # Single character
    if len(token) == 1:
        return TokenType.SINGLE_CHARACTER

    # Check if token is in our set of known words
    token_without_space = token.rstrip()

    # Full word check (using our word set for validation)
    if token in word_set:
        return TokenType.FULL_WORD

    # Word followed by space
    if token.endswith(" ") and token_without_space in word_set:
        return TokenType.WORD_FOLLOWED_BY_SPACE

    # Check if it consists only of letters (but not a known word)
    if token.isalpha():
        return TokenType.NGRAM_LETTERS_ONLY

    # Check if it's an n-gram with no letters
    if not any(c.isalpha() for c in token):
        return TokenType.NGRAM_NO_LETTERS

    # Default case
    return TokenType.OTHER


def set_word_set_for_cache(word_set: set[str]) -> None:
    """Set the global word set for classification.

    Args:
        word_set: The set of words to use for classification
    """
    global _word_set_for_cache
    _word_set_for_cache = word_set


def extract_tokens_from_segmentation(
    segmentation: List[TextSegment],
    min_token_length: int,
    max_token_length: int,
) -> List[TokenData]:
    """Extract all token candidates from current optimal text segmentation.

    Args:
        segmentation: List of TextSegment objects from optimal segmentation
        min_token_length: Minimum subtoken count in a token
        max_token_length: Maximum subtoken count in a token

    Returns:
        List of extracted TokenData objects
    """
    # Counter for token frequencies
    token_counter = Counter()
    token_compositions = {}

    # Sliding window on segmented positions
    text_current_subtoken_count = len(segmentation)

    # Iterate over the text with each allowed window size
    for window_size in range(min_token_length, max_token_length + 1):
        for window_position in range(text_current_subtoken_count - window_size + 1):
            # Extract current window of tokens
            window = segmentation[window_position : window_position + window_size]

            # Store the composition (which selected tokens make up this new token)
            composition = [segment.token_text for segment in window]

            # Get text span
            token_text = "".join(composition)

            # Count this token
            token_counter[token_text] += 1

            # Track token_composition and assert there are no conflicts
            if token_text in token_compositions:
                assert token_compositions[token_text] == composition
            else:
                token_compositions[token_text] = composition

    # Create TokenData objects
    token_data_list = []

    for token_text, count in token_counter.items():
        # Classify token
        token_type = classify_token(token_text)

        # Create TokenData
        token_data = TokenData(
            lower=token_text,
            character_length=len(token_text),
            subtoken_length=len(token_compositions.get(token_text, [token_text])),
            token_type=token_type,
            text_count=count,
            usage_count=count,
            rank=0,  # Will be assigned later
            usage_cost=0.0,  # Will be calculated later
            replacement_score=0.0,  # Will be calculated later
            selected=False,
            best_current_combination=token_compositions.get(token_text, [token_text]),
            adjacent_tokens=None,
        )

        token_data_list.append(token_data)

    return token_data_list


def process_chunk_for_parallel(args):
    """Process a chunk of segmentation for parallel token extraction.

    Args:
        args: Tuple containing (chunk, min_token_length, max_token_length, word_set)

    Returns:
        List of extracted TokenData objects
    """
    chunk, min_token_length, max_token_length, word_set = args
    # Set the word set for this worker process
    set_word_set_for_cache(word_set)
    # Process this chunk using the original function
    return extract_tokens_from_segmentation(chunk, min_token_length, max_token_length)


def extract_tokens_from_segmentation_parallel(
    segmentation: List[TextSegment],
    min_token_length: int,
    max_token_length: int,
) -> List[TokenData]:
    """Extract new token candidates from optimal text segmentation using parallel processing.

    Args:
        segmentation: List of TextSegment objects
        min_token_length: Minimum token count
        max_token_length: Maximum token count

    Returns:
        List of extracted TokenData objects
    """

    # For small segmentations, use sequential approach
    if len(segmentation) < 1000:
        return extract_tokens_from_segmentation(
            segmentation, min_token_length, max_token_length
        )

    # Get the word set once and copy it
    word_set = _word_set_for_cache.copy()

    # Split work into chunks for parallel processing
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(100, len(segmentation) // cpu_count)

    # Create work chunks with overlap to ensure we don't miss tokens at boundaries
    chunks = []
    n = len(segmentation)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size + max_token_length - 1, n)
        chunks.append(segmentation[i:end])

    # Prepare arguments for the parallel function
    args = [(chunk, min_token_length, max_token_length, word_set) for chunk in chunks]

    # Process chunks in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(process_chunk_for_parallel, args)

    # Combine results with composition validation
    all_tokens = {}

    for chunk_tokens in results:
        for token in chunk_tokens:
            key = token.lower
            if key in all_tokens:
                # Validate that compositions match
                assert (
                    token.best_current_combination
                    == all_tokens[key].best_current_combination
                ), f"Composition mismatch for token '{key}'"

                # Merge frequencies
                all_tokens[key].text_count += token.text_count
                all_tokens[key].usage_count += token.usage_count
            else:
                all_tokens[key] = token

    return list(all_tokens.values())
