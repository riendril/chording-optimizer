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
from typing import Any, Dict, List, Optional, Set, Tuple

import tqdm

from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData, TokenType

logger = logging.getLogger(__name__)


# Data structure for interval tracking
class IntervalSet:
    """Efficient data structure for tracking intervals in a text

    This is used to avoid double-counting tokens during extraction.
    """

    def __init__(self):
        self.intervals = []  # List of (start, end) tuples

    def add_interval(self, start: int, end: int) -> bool:
        """Add an interval if it doesn't overlap with existing intervals

        Args:
            start: Start position of interval
            end: End position of interval

        Returns:
            True if interval was added, False if it overlaps with existing interval
        """
        # Check for overlaps using binary search (O(log n))
        idx = self._find_insertion_index(start)

        # Check if this interval overlaps with previous interval
        if idx > 0 and self.intervals[idx - 1][1] > start:
            return False

        # Check if this interval overlaps with next interval
        if idx < len(self.intervals) and self.intervals[idx][0] < end:
            return False

        # Insert the interval at the correct position
        self.intervals.insert(idx, (start, end))
        return True

    def _find_insertion_index(self, start: int) -> int:
        """Binary search to find insertion index for a new interval

        Args:
            start: Start position of interval

        Returns:
            Index where the interval should be inserted
        """
        left, right = 0, len(self.intervals)

        while left < right:
            mid = (left + right) // 2
            if self.intervals[mid][0] < start:
                left = mid + 1
            else:
                right = mid

        return left


# Trie node for efficient subtoken/supertoken detection
class TrieNode:
    """Node in a trie data structure for efficient string operations"""

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.token_data = None
        self.indices = []  # Store positions where this node ends a word


class Trie:
    """Trie data structure for efficient subtoken/supertoken detection"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_data: TokenData, index: int):
        """Insert a token into the trie

        Args:
            token_data: TokenData object to insert
            index: Position of this token in the original list
        """
        node = self.root
        for char in token_data.lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.token_data = token_data
        node.indices.append(index)

    def find_subtokens(self, token: str) -> List[Tuple[int, TokenData]]:
        """Find all subtokens within a token

        Args:
            token: Token to search for subtokens

        Returns:
            List of (index, TokenData) pairs for all subtokens
        """
        results = []

        # For each starting position in the token
        for i in range(len(token)):
            node = self.root
            # For each possible ending position from this start
            for j in range(i, len(token)):
                char = token[j]
                if char not in node.children:
                    break

                node = node.children[char]
                # If this is a complete token (not just a prefix)
                if node.is_end_of_word and node.token_data.lower != token:
                    for idx in node.indices:
                        results.append((idx, node.token_data))

        return results


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

    # Extract tokens with efficient non-overlapping algorithm
    extract_tokens_efficiently(chunk, min_length, max_length, result)

    return result


def extract_tokens_efficiently(
    text: str, min_length: int, max_length: int, token_counts: Counter
):
    """Extract tokens from text with O(n log n) efficiency, avoiding double-counting

    Args:
        text: Text to extract tokens from
        min_length: Minimum token length to include
        max_length: Maximum token length to include
        token_counts: Counter to update with tokens
    """
    # Use IntervalSet to track which parts of text have been processed
    intervals = IntervalSet()

    # Priority order for token types (highest to lowest priority)
    # 1. Words followed by space
    # 2. Complete words
    # 3. Word n-grams
    # 4. Character n-grams

    # First extract and track word positions (for classification)
    word_positions = []
    for match in re.finditer(r"\b\w+\b", text):
        start, end = match.span()
        word = match.group()
        word_positions.append((start, end, word))

        # If word meets length criteria, count it and mark its interval
        if min_length <= len(word) <= max_length:
            token_counts[word] += 1
            intervals.add_interval(start, end)

        # Check for "word followed by space"
        if end < len(text) and text[end] == " ":
            word_space = word + " "
            if min_length <= len(word_space) <= max_length:
                token_counts[word_space] += 1
                intervals.add_interval(start, end + 1)  # Include the space

    # Process word n-grams efficiently
    words = [word for _, _, word in word_positions]
    for n in range(2, 4):  # 2-3 word phrases
        for i in range(len(words) - n + 1):
            # Find the start and end positions of this n-gram
            start_pos = word_positions[i][0]
            end_pos = word_positions[i + n - 1][1]

            # Only process if this n-gram fits the length criteria
            ngram = " ".join(words[i : i + n])
            if min_length <= len(ngram) <= max_length:
                # Check if this interval has already been covered
                if intervals.add_interval(start_pos, end_pos):
                    token_counts[ngram] += 1

    # Process character n-grams that haven't been covered
    # We'll process in small chunks for better performance
    chunk_size = 100
    for chunk_start in range(0, len(text), chunk_size):
        chunk_end = min(chunk_start + chunk_size + max_length, len(text))
        chunk = text[chunk_start:chunk_end]

        # For each possible n-gram length
        for n in range(min_length, max_length + 1):
            # For each possible starting position
            for i in range(len(chunk) - n + 1):
                abs_start = chunk_start + i
                abs_end = abs_start + n

                # Only process if this interval hasn't been covered yet
                if intervals.add_interval(abs_start, abs_end):
                    ngram = chunk[i : i + n]
                    # Skip n-grams with non-printable chars or excessive whitespace
                    if all(c in string.printable for c in ngram) and not re.search(
                        r"\s{2,}", ngram
                    ):
                        token_counts[ngram] += 1


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

    # Apply subtoken/supertoken adjustments based on learning limit
    logger.info("Applying subtoken/supertoken adjustments")
    benchmark.start_phase(BenchmarkPhase.SET_IMPROVEMENT)
    final_tokens = apply_token_adjustments_with_limit(
        initial_top_tokens,
        token_counts,
        config.token_analysis.learning_limit_type,
        benchmark,
    )
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

    logger.info("Extracting tokens (efficient non-overlapping algorithm)...")
    # Use efficient non-overlapping extraction
    extract_tokens_efficiently(text, min_length, max_length, token_counts)
    benchmark.update_phase(len(token_counts))

    return token_counts


def _extract_tokens_parallel(
    text: str, min_length: int, max_length: int, benchmark: Benchmark
) -> Dict[str, int]:
    """Extract tokens using parallel processing"""
    # Split text into chunks
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(1000, len(text) // cpu_count)

    # Add overlap to ensure we don't miss tokens at boundaries
    overlap = max_length - 1

    # Create overlapping chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_end = min(i + chunk_size + overlap, len(text))
        chunks.append(text[i:chunk_end])

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


def apply_token_adjustments_with_limit(
    token_collection: TokenCollection,
    token_counts: Dict[str, int],
    learning_limit: TokenType,
    benchmark: Benchmark,
) -> TokenCollection:
    """Apply subtoken/supertoken adjustments using a trie, respecting learning limit

    Args:
        token_collection: Initial collection of top tokens
        token_counts: Complete dictionary of token counts
        learning_limit: Maximum token type complexity to consider
        benchmark: Benchmark instance for performance tracking

    Returns:
        Adjusted token collection
    """
    # Make a copy of the original token list
    tokens = token_collection.tokens.copy()
    total_count = sum(token_counts.values())

    # Filter tokens based on learning limit
    logger.info(f"Filtering tokens based on learning limit: {learning_limit.name}")
    eligible_tokens = [t for t in tokens if t.token_type <= learning_limit]

    # Log statistics about eligible tokens
    token_type_counts = {t.name: 0 for t in TokenType}
    for t in tokens:
        token_type_counts[t.token_type.name] += 1

    logger.info(f"Token type distribution: {token_type_counts}")
    logger.info(
        f"Eligible tokens for adjustment: {len(eligible_tokens)} out of {len(tokens)}"
    )

    # Build trie for efficient subtoken detection
    logger.info("Building trie for token relationships...")
    trie = Trie()
    for i, token in enumerate(eligible_tokens):
        trie.insert(token, i)

    # Process tokens in order of score (highest first)
    logger.info("Processing tokens with trie-based algorithm...")
    processed = set()

    # Process in order of initial ranking, but only for eligible tokens
    tokens_by_score = sorted(eligible_tokens, key=lambda t: t.score, reverse=True)

    for token in tqdm.tqdm(tokens_by_score, desc="Adjusting token scores"):
        if token.lower in processed:
            continue

        processed.add(token.lower)

        # Find all subtokens using trie (much more efficient)
        subtokens = trie.find_subtokens(token.lower)
        for idx, subtoken in subtokens:
            if subtoken.lower in processed:
                continue

            # Adjust frequency and score
            adjusted_freq = subtoken.frequency - token.frequency
            subtoken.frequency = adjusted_freq
            subtoken.score = (adjusted_freq / total_count) * len(subtoken.lower)

        # Check if this token is a subtoken of any other token
        for other_token in eligible_tokens:
            if other_token.lower in processed or other_token.lower == token.lower:
                continue

            if token.lower in other_token.lower:
                # If this token is contained in the other token
                effective_length = max(
                    1, len(other_token.lower) - (len(token.lower) - 1)
                )
                other_token.score = (
                    other_token.frequency / total_count
                ) * effective_length

        benchmark.update_phase(len(processed))

    # Re-sort all tokens by adjusted score
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
