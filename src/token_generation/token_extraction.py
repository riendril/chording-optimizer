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
from typing import List, Tuple

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


def classify_token(token: str, word_set: set[str]) -> TokenType:
    """Classify a token into its type category using word context.

    Args:
        token: The token string to classify
        word_set: set of known words from the corpus

    Returns:
        TokenType enumeration value
    """
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
        return TokenType.WORD_WITH_SPACE

    # Check if it consists only of letters (but not a known word)
    if token.isalpha():
        return TokenType.NGRAM_LETTERS_ONLY

    # Check if it's an n-gram with no letters
    if not any(c.isalpha() for c in token):
        return TokenType.NGRAM_NO_LETTERS

    # Default case
    return TokenType.OTHER


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

    # TODO: Check if this is legit

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
                    for idx in node.indices:
                        results.append((idx, node.token_data))

        return results

    def find_supertokens(
        self, token: str, collection: TokenCollection
    ) -> List[TokenData]:
        """Find all supertokens that contain this token as a substring

        Args:
            token: Token to search for as a substring
            collection: TokenCollection containing all tokens to search within

        Returns:
            List of TokenData objects for tokens that contain the input token
        """
        return [
            other_token
            for other_token in collection.tokens
            if token in other_token.lower and other_token.lower != token
        ]


def extract_tokens_sliding_window(
    text: str,
    min_length: int,
    max_length: int,
    token_collection: TokenCollection,
    word_set: set[str],
):
    """Extract all tokens using a sliding window approach with classification

    Args:
        text: Text to extract tokens from
        min_length: Minimum token length to include
        max_length: Maximum token length to include
        token_collection: TokenCollection to update with tokens
        word_set: set of known words from the corpus
    """
    # Create a temporary counter for frequency tracking
    temp_counter = Counter()

    # Step 1: Sliding window extraction for all possible tokens
    for length in range(min_length, max_length + 1):
        for i in range(len(text) - length + 1):
            end = i + length
            token = text[i:end]
            # Count the token
            temp_counter[token] += 1

    # Step 2: Convert to TokenData objects and update collection
    for token, count in temp_counter.items():
        # Check if token already exists in collection
        existing_tokens = [
            t for t in token_collection.tokens if t.lower == token.lower()
        ]
        assert len(existing_tokens) <= 1
        if existing_tokens:
            # Update count of existing token
            existing_tokens[0].count += count
        else:
            # Create new TokenData object directly
            token_data = TokenData(
                lower=token.lower(),
                length=len(token),
                token_type=classify_token(token, word_set),
                count=count,
                rank=0,  # Rank will be assigned later
                score=0.0,  # Score will be calculated later
                selected=False,
            )
            token_collection.tokens.append(token_data)


# Function defined at module level for multiprocessing compatibility
def process_chunk(chunk, min_length, max_length, word_set):
    """Process a single text chunk for token extraction.

    Args:
        chunk: Text chunk to process
        min_length: Minimum token length to include
        max_length: Maximum token length to include
        word_set: set of known words from the corpus

    Returns:
        TokenCollection with tokens extracted from the chunk
    """
    token_collection = TokenCollection(
        name="chunk_extraction",
        tokens=[],
        ordered_by_frequency=False,
        source="chunk_processing",
    )

    # Extract tokens using sliding window approach
    extract_tokens_sliding_window(
        chunk, min_length, max_length, token_collection, word_set
    )

    return token_collection


def _extract_tokens_parallel(
    text: str,
    min_length: int,
    max_length: int,
    word_set: set[str],
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
        # Process each chunk with progress indication
        results = []

        for chunk in chunks:
            results.append(
                pool.apply_async(
                    process_chunk, args=(chunk, min_length, max_length, word_set)
                )
            )

        # Collect results with progress bar
        combined_collection = TokenCollection(
            name="parallel_extraction",
            tokens=[],
            ordered_by_frequency=False,
            source="parallel_processing",
        )

        for result in tqdm.tqdm(results, desc="Processing chunks"):
            chunk_collection = result.get()

            # Merge chunk's tokens into the combined collection
            for token_data in chunk_collection.tokens:
                existing_token = next(
                    (
                        t
                        for t in combined_collection.tokens
                        if t.lower == token_data.lower
                    ),
                    None,
                )

                if existing_token:
                    # Update count of existing token
                    existing_token.count += token_data.count
                else:
                    # Add new token to collection
                    combined_collection.tokens.append(token_data)

            benchmark.update_phase(len(combined_collection.tokens))

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

    # Process in parallel or sequentially
    if use_parallel and len(text) > 100000:  # Only parallelize for large corpora
        return _extract_tokens_parallel(
            text, min_length, max_length, word_set, benchmark
        )

    logger.info("Extracting tokens using sliding window approach...")
    # Use sliding window extraction
    extract_tokens_sliding_window(
        text, min_length, max_length, token_collection, word_set
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
    # Find total token count for normalization
    total_count = sum(token.count for token in token_collection.tokens)

    # Calculate scores
    for token in tqdm.tqdm(token_collection.tokens, desc="Scoring tokens"):
        # Base score is normalized frequency
        frequency_score = token.count / total_count

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
            count=t.count,
            rank=t.rank,
            score=t.score,
            selected=t.selected,
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
    for i, token in enumerate(token_list):
        token_trie.insert(token, i)

    # Create a new collection for selection and adjustment
    temp_collection = TokenCollection(
        name=token_collection.name,
        tokens=token_list,
        ordered_by_frequency=True,
        source=token_collection.source,
    )

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
        selected_count += 1

        # Find subtokens of this token
        subtokens = token_trie.find_subtokens(token.lower)
        for _, subtoken in subtokens:
            # Adjust count and score of subtokens
            subtoken.count -= token.count
            subtoken.score = (subtoken.count / len(token_list)) * subtoken.length

        # Find supertokens that contain this token
        supertokens = token_trie.find_supertokens(token.lower, temp_collection)
        # Do nothing with them for now
        # TODO: This function needs to take into account supertokens that
        # contain the token multiple times somehow

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
