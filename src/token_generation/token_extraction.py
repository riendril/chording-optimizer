"""
Token extraction, counting, and scoring module.

This module processes corpus text to:
1. Extract all possible tokens within min/max length constraints using a sliding window approach
2. Score tokens based on frequency and length
3. Select tokens to be assigned to chords while making subtoken/supertoken adjustments
4. Output a ranked list of tokens for chord assignment
"""

import logging
import multiprocessing
import os
import re
from collections import Counter
from functools import lru_cache
from typing import List

import tqdm

from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.shared_types import TokenCollection, TokenData, TokenType

logger = logging.getLogger(__name__)


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


# Global variable to hold the word set for caching
_word_set_for_cache = set()


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


# Trie node for efficient subtoken/supertoken detection
class TrieNode:
    """Node in a trie data structure for efficient string operations"""

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.token_data = None


class Trie:
    """Trie data structure for efficient subtoken/supertoken detection"""

    # TODO: Check if this is legit

    def __init__(self):
        self.root = TrieNode()
        self.token_to_supertokens = {}  # For efficient supertoken lookup

    def insert(self, token_data: TokenData):
        """Insert a token into the trie and update supertoken relationships

        Args:
            token_data: TokenData object to insert
        """
        node = self.root
        token = token_data.lower

        # Insert token into trie (standard Trie insertion)
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.token_data = token_data

        # Record this token as a supertoken for all its substrings
        self._update_supertoken_relationships(token_data)

    def _update_supertoken_relationships(self, supertoken_data: TokenData):
        """Register a token as a supertoken for all its substrings

        Args:
            supertoken_data: TokenData object to register as a supertoken
        """
        token = supertoken_data.lower
        token_len = len(token)

        # For each position in the token
        for i in range(token_len):
            # For each possible substring starting at this position
            for j in range(i + 1, token_len + 1):
                if j - i > 1:  # Only consider meaningful substrings (length > 1)
                    substring = token[i:j]

                    # Add this token as a supertoken of this substring
                    if substring not in self.token_to_supertokens:
                        self.token_to_supertokens[substring] = []

                    # Avoid duplicates
                    if supertoken_data not in self.token_to_supertokens[substring]:
                        self.token_to_supertokens[substring].append(supertoken_data)

    def find_subtokens(self, token: str) -> List[TokenData]:
        """Find all subtokens within a token

        Args:
            token: Token to search for subtokens

        Returns:
            List of TokenData objects for all subtokens found
        """
        results = []
        # TODO: check if this function is legit

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
                    results.append(node.token_data)

        return results

    def find_supertokens(self, token: str) -> List[TokenData]:
        """Find all supertokens that contain this token as a substring

        Args:
            token: Token to search for as a substring

        Returns:
            List of TokenData objects for tokens that contain the input token
        """
        if token in self.token_to_supertokens:
            return self.token_to_supertokens[token]
        return []


def extract_tokens_sliding_window(
    text: str,
    min_length: int,
    max_length: int,
    token_collection: TokenCollection,
    word_set_id: int,
    pbar=None,
):
    """Extract all tokens using a sliding window approach with classification

    Args:
        text: Text to extract tokens from
        min_length: Minimum token length to include
        max_length: Maximum token length to include
        token_collection: TokenCollection to update with tokens
        word_set_id: Identifier for the word set to use in classification
        pbar: Optional progress bar to update
    """
    # Create a temporary counter for frequency tracking
    temp_counter = Counter()

    # Calculate total operations for progress reporting
    total_ops = sum(
        len(text) - length + 1 for length in range(min_length, max_length + 1)
    )

    # Step 1: Sliding window extraction for all possible tokens
    # Process by length for better cache locality
    for length in range(min_length, max_length + 1):
        # Extract tokens of this length
        length_tokens = [text[i : i + length] for i in range(len(text) - length + 1)]

        # Update counter in batch
        temp_counter.update(length_tokens)

        # Update progress bar if provided
        if pbar:
            pbar.update(len(length_tokens))

    # Step 2: Convert to TokenData objects and update collection
    # Batch token classification for better performance
    unique_tokens = list(temp_counter.keys())

    # Process tokens in batches
    token_data_list = []
    for token in unique_tokens:
        count = temp_counter[token]
        token_type = classify_token(token, word_set_id)

        # Create TokenData object
        token_data = TokenData(
            lower=token.lower(),
            length=len(token),
            token_type=token_type,
            text_count=count,
            usage_count=count,
            rank=0,  # Rank will be assigned later
            score=0.0,  # Score will be calculated later
            selected=False,
            best_current_combination=list(
                token.lower()
            ),  # Initialize with single characters
        )
        token_data_list.append(token_data)

    # Update collection with all tokens at once
    token_collection.tokens.extend(token_data_list)


# Function defined at module level for multiprocessing compatibility
def process_chunk(chunk_data):
    """Process a single text chunk for token extraction.

    Args:
        chunk_data: Tuple of (chunk, min_length, max_length, word_set_id, chunk_id)

    Returns:
        TokenCollection with tokens extracted from the chunk
    """
    chunk, min_length, max_length, word_set_id, chunk_id = chunk_data

    token_collection = TokenCollection(
        name="chunk_extraction",
        tokens=[],
        ordered_by_frequency=False,
        source="chunk_processing",
    )

    # Create a progress bar for this chunk
    total_ops = sum(
        len(chunk) - length + 1 for length in range(min_length, max_length + 1)
    )

    # Extract tokens using sliding window approach (without progress bar in worker)
    extract_tokens_sliding_window(
        chunk, min_length, max_length, token_collection, word_set_id
    )

    return token_collection


def _extract_tokens_parallel(
    text: str,
    min_length: int,
    max_length: int,
    word_set_id: int,
    benchmark: Benchmark,
) -> TokenCollection:
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
        # Process each chunk
        results = []

        # Create work items
        work_items = [
            (chunk, min_length, max_length, word_set_id, i)
            for i, chunk in enumerate(chunks)
        ]

        # Start processing asynchronously
        for item in work_items:
            results.append(pool.apply_async(process_chunk, args=(item,)))

        # Create a progress bar for tracking chunks completion
        with tqdm.tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            # Poll for completed chunks
            completed = [False] * len(chunks)
            while not all(completed):
                for i, result in enumerate(results):
                    if not completed[i] and result.ready():
                        completed[i] = True
                        pbar.update(1)

                # Don't busy-wait
                if not all(completed):
                    import time

                    time.sleep(0.1)

        # Collect and merge results
        combined_collection = TokenCollection(
            name="parallel_extraction",
            tokens=[],
            ordered_by_frequency=False,
            source="parallel_processing",
        )

        # Merge token data
        all_tokens = {}

        # Show a progress bar for merging results
        with tqdm.tqdm(total=len(results), desc="Merging results") as pbar:
            for i, result in enumerate(results):
                chunk_collection = result.get()

                # Update token counts
                for token_data in chunk_collection.tokens:
                    token = token_data.lower
                    if token in all_tokens:
                        all_tokens[token].text_count += token_data.text_count
                        all_tokens[token].usage_count += token_data.text_count
                    else:
                        all_tokens[token] = token_data

                benchmark.update_phase(len(all_tokens))
                pbar.update(1)

        # Convert merged tokens to list
        combined_collection.tokens = list(all_tokens.values())

    return combined_collection


def extract_all_tokens(
    text: str,
    min_length: int,
    max_length: int,
    use_parallel: bool,
    benchmark: Benchmark,
) -> TokenCollection:
    """Extract all possible tokens within length constraints

    Args:
        text: Input corpus text
        min_length: Minimum token length to include
        max_length: Maximum token length to include
        use_parallel: Whether to use parallel processing
        benchmark: Benchmark instance for performance tracking

    Returns:
        TokenCollection with all extracted tokens
    """
    # Create a new token collection
    token_collection = TokenCollection(
        name="corpus_extraction",
        tokens=[],
        ordered_by_frequency=False,
        source="extract_all_tokens",
    )

    # Extract words from the text for linguistic context
    logger.info("Extracting words from corpus using word boundaries")
    word_set = extract_words_from_text(text)
    logger.info(f"Extracted {len(word_set)} unique words from corpus")

    # Set global word set for caching and get identifier
    word_set_id = set_word_set_for_cache(word_set)

    # Process in parallel or sequentially
    if use_parallel and len(text) > 100000:  # Only parallelize for large corpora
        return _extract_tokens_parallel(
            text, min_length, max_length, word_set_id, benchmark
        )

    logger.info("Extracting tokens using sliding window approach...")

    # Calculate total operations for progress reporting
    total_ops = sum(
        len(text) - length + 1 for length in range(min_length, max_length + 1)
    )

    # Use sliding window extraction with progress bar
    with tqdm.tqdm(total=total_ops, desc="Extracting tokens") as pbar:
        extract_tokens_sliding_window(
            text, min_length, max_length, token_collection, word_set_id, pbar
        )

    benchmark.update_phase(len(token_collection.tokens))

    return token_collection


def score_tokens(token_collection: TokenCollection) -> TokenCollection:
    """Score tokens based on frequency and length

    Args:
        token_collection: TokenCollection with tokens to score

    Returns:
        TokenCollection with updated scores
    """
    # TODO: Do normalization with just text length instead and pass it to this
    # function as an int

    # Find total token count for normalization
    total_count = sum(token.text_count for token in token_collection.tokens)

    # Calculate scores
    for token in tqdm.tqdm(token_collection.tokens, desc="Scoring tokens"):
        # Base score is normalized frequency
        frequency_score = token.text_count / total_count

        # Multiply by length for length benefit
        token.score = frequency_score * token.length

    return token_collection


def select_tokens_and_adjust(
    token_collection: TokenCollection,
    top_n: int,
    learning_limit: TokenType,
    chords_to_assign: int,
    benchmark: Benchmark,
) -> TokenCollection:
    """Apply subtoken/supertoken adjustments using a trie, respecting learning limit

    Args:
        token_collection: Collection containing all tokens
        top_n: Number of top tokens to consider
        learning_limit: Maximum token type complexity to consider
        chords_to_assign: Number of tokens/chords to assign
        benchmark: Benchmark instance for performance tracking

    Returns:
        Adjusted token collection
    """
    # Sort tokens by score if they aren't already
    sorted_tokens = sorted(token_collection.tokens, key=lambda t: t.score, reverse=True)

    # Take top N tokens only
    top_tokens = sorted_tokens[:top_n]

    # set ranks for the top tokens
    for i, token in enumerate(top_tokens):
        token.rank = i + 1

    # Make a deep copy of the top tokens
    token_list = [
        TokenData(
            lower=t.lower,
            length=t.length,
            token_type=t.token_type,
            text_count=t.text_count,
            usage_count=t.usage_count,
            rank=t.rank,
            score=t.score,
            selected=t.selected,
            best_current_combination=list(t.best_current_combination),
        )
        for t in top_tokens
    ]

    # Filter out eligible tokens based on learning limit
    logger.info(f"Filtering tokens based on learning limit: {learning_limit.name}")
    eligible_tokens = sorted(
        [t for t in token_list if t.token_type <= learning_limit],
        key=lambda t: t.score,
        reverse=True,
    )

    # Log statistics about eligible tokens
    token_type_counts = {t.name: 0 for t in TokenType}
    for t in token_list:
        token_type_counts[t.token_type.name] += 1

    logger.info(f"Token type distribution: {token_type_counts}")
    logger.info(
        f"Eligible tokens for adjustment: {len(eligible_tokens)} out of {len(token_list)}"
    )

    # Build trie for efficient token relationship detection
    logger.info("Building trie for token relationships...")
    token_trie = Trie()
    for token in token_list:
        token_trie.insert(token)

    # Select tokens while adjusting related tokens
    logger.info("Selecting and rescoring tokens")
    selected_count = 0

    # While there are less than the configured amount of desired chords selected:
    for _ in tqdm.tqdm(range(chords_to_assign), desc="Selecting tokens"):
        if not eligible_tokens or selected_count >= chords_to_assign:
            break

        # Select the current highest not yet selected token
        token = eligible_tokens.pop(0)
        token.selected = True
        token.best_current_combination = [token.lower]
        selected_count += 1

        # Find subtokens of this token
        subtokens = token_trie.find_subtokens(token.lower)
        for subtoken in subtokens:
            # Adjust usage_count and score of subtokens
            subtoken.usage_count -= token.usage_count
            subtoken.score = (subtoken.usage_count / len(token_list)) * subtoken.length

        # Find supertokens that contain this token
        supertokens = token_trie.find_supertokens(token.lower)
        for supertoken in supertokens:
            occurrences = supertoken.lower.count(token.lower)
            # Do nothing with them for now
            # TODO: Adjust supertokens
            # account for multiple containments

        # TODO: Adjust overlapping tokens

        # TODO: Update best_current_combination for all tokens

        # Resort tokens by adjusted score
        eligible_tokens.sort(
            key=lambda t: t.score,
            reverse=True,
        )

        benchmark.update_phase(selected_count)

    # Re-sort all tokens by adjusted score and selection status
    token_list.sort(key=lambda t: (-1 if t.selected else 0, t.score), reverse=True)

    # Update ranks
    for i, token in enumerate(token_list):
        token.rank = i + 1

    # Log selection statistics
    selected = [t for t in token_list if t.selected]
    logger.info(f"Selected {len(selected)} tokens out of {len(token_list)}")

    # Create new collection with adjusted tokens
    return TokenCollection(
        name=f"{token_collection.name}_adjusted",
        tokens=token_list,
        ordered_by_frequency=True,
        source=token_collection.source,
    )


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
    token_collection = extract_all_tokens(
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
    token_collection = score_tokens(token_collection)
    benchmark.end_phase()

    # Apply subtoken/supertoken adjustments based on learning limit
    logger.info(
        f"Selecting and adjusting top {config.token_analysis.top_n_tokens} tokens"
    )
    benchmark.start_phase(BenchmarkPhase.SET_IMPROVEMENT)
    final_tokens = select_tokens_and_adjust(
        token_collection,
        config.token_analysis.top_n_tokens,
        config.token_analysis.learning_limit_type,
        config.general.chords_to_assign,
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
