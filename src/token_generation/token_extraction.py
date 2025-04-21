"""
Token extraction module for chord optimization.

This module handles extracting tokens and building contextual relationships
in a single efficient pass, supporting various token types and optimizations.
"""

import re
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from src.common.config import GeneratorConfig
from src.common.shared_types import ContextInfo, TokenCollection, TokenData

# -----------------
# Text Preprocessing
# -----------------


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
# Context Building Functions
# -----------------


def process_text_for_context(
    text: str, tokens: Set[str], window_size: int
) -> Dict[str, Dict]:
    """
    Process text to extract contextual relationships between tokens.

    Args:
        text: Preprocessed text
        tokens: Set of tokens to track context for
        window_size: Size of window for preceding/following tokens

    Returns:
        Dictionary with preceding and following relationship counts
    """
    context_data = {
        token: {"preceding": Counter(), "following": Counter()} for token in tokens
    }

    # Find token contexts in paragraphs
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        words = re.findall(r"\b[\w\']+\b", paragraph.lower())
        if len(words) <= 1:
            continue

        for i, word in enumerate(words):
            if word not in tokens:
                continue

            # Find preceding tokens
            for offset in range(1, window_size + 1):
                if i - offset >= 0:
                    prev_word = words[i - offset]
                    if prev_word in tokens:
                        context_data[word]["preceding"][prev_word] += 1

            # Find following tokens
            for offset in range(1, window_size + 1):
                if i + offset < len(words):
                    next_word = words[i + offset]
                    if next_word in tokens:
                        context_data[word]["following"][next_word] += 1

    return context_data


def find_substring_relationships(
    tokens: Dict[str, int],
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Find substring relationships between tokens.

    Args:
        tokens: Dictionary of tokens and their frequencies

    Returns:
        Dictionary mapping tokens to (is_substring_of, contains_substrings) lists
    """
    tokens_list = list(tokens.keys())
    tokens_list.sort(key=len)  # Sort by length to find substrings efficiently

    substring_relationships = {token: ([], []) for token in tokens_list}

    for i, shorter in enumerate(tokens_list):
        if len(shorter) <= 1:
            continue  # Skip single-character tokens

        for longer in tokens_list[i + 1 :]:
            if shorter in longer:
                # shorter is substring of longer
                substring_relationships[shorter][0].append(longer)
                # longer contains shorter
                substring_relationships[longer][1].append(shorter)

    return substring_relationships


def build_context_information(
    text: str, tokens: Dict[str, int], window_size: int = 2, min_freq: int = 2
) -> Dict[str, ContextInfo]:
    """
    Extract context information for tokens including preceding/following
    relationships and substring relationships.

    Args:
        text: Preprocessed text
        tokens: Dictionary mapping tokens to their frequencies
        window_size: Size of window to look for preceding/following tokens
        min_freq: Minimum frequency to include a context relationship

    Returns:
        Dictionary mapping tokens to their ContextInfo objects
    """
    # Process text for preceding/following relationships
    token_set = set(tokens.keys())
    context_data = process_text_for_context(text, token_set, window_size)

    # Find substring relationships
    substring_data = find_substring_relationships(tokens)

    # Create ContextInfo objects
    context_info = {}
    for token in tokens:
        # Filter low-frequency relationships
        preceding = {
            t: freq
            for t, freq in context_data[token]["preceding"].items()
            if freq >= min_freq
        }

        following = {
            t: freq
            for t, freq in context_data[token]["following"].items()
            if freq >= min_freq
        }

        is_substring_of, contains_substrings = substring_data.get(token, ([], []))

        context_info[token] = ContextInfo(
            preceding=preceding,
            following=following,
            is_substring_of=is_substring_of,
            contains_substrings=contains_substrings,
        )

    return context_info


# -----------------
# Top-Level Functions
# -----------------


def extract_tokens_with_context(
    corpus: str,
    config: GeneratorConfig,
    window_size: int = 2,
    min_freq: int = 2,
    show_progress: bool = True,
) -> Tuple[Dict[str, int], Dict[str, ContextInfo]]:
    """
    Extract tokens and build context information in a single efficient pass.

    Args:
        corpus: Raw input text
        config: Generator configuration
        window_size: Size of window for context relationships
        min_freq: Minimum frequency for context relationships
        show_progress: Whether to show progress updates

    Returns:
        Tuple of (token frequencies, context information)
    """
    if show_progress:
        print("Preprocessing text...")

    processed_text = preprocess_text(corpus)

    if show_progress:
        print("Extracting tokens...")

    tokens = extract_tokens(processed_text, config)

    if show_progress:
        print(f"Found {len(tokens)} unique tokens. Building context information...")

    context_info = build_context_information(
        processed_text, tokens, window_size, min_freq
    )

    if show_progress:
        print(f"Built context information for {len(context_info)} tokens.")

    return tokens, context_info


def create_token_collection_with_context(
    tokens: Dict[str, int],
    name: str,
    source: Optional[str] = None,
    context_info: Optional[Dict[str, ContextInfo]] = None,
    zipf_weight_base: float = 1.0,
) -> TokenCollection:
    """
    Create a TokenCollection with optional context information.

    Args:
        tokens: Dictionary mapping tokens to their frequencies
        name: Name for the collection
        source: Source identifier for the collection
        context_info: Dictionary mapping tokens to their context information
        zipf_weight_base: Base value for Zipf weighting

    Returns:
        TokenCollection object with token and context data
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

        # Add context if available
        if context_info and token in context_info:
            token_data.context = context_info[token]

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


# Main API function for complete token extraction and collection creation
def analyze_corpus_with_context(
    corpus: str,
    config: GeneratorConfig,
    top_n: Optional[int] = None,
    extract_context: bool = True,
    context_window_size: int = 2,
    context_min_freq: int = 2,
    show_progress: bool = True,
) -> Tuple[Dict[str, int], Dict[str, ContextInfo]]:
    """
    Complete pipeline for corpus analysis with context extraction.

    Args:
        corpus: Raw input text
        config: Generator configuration
        top_n: Number of top tokens to keep (by frequency)
        extract_context: Whether to extract context information
        context_window_size: Size of window for context relationships
        context_min_freq: Minimum frequency for context relationships
        show_progress: Whether to show progress updates

    Returns:
        Tuple of (token frequencies, context information)
    """
    # Extract tokens and context
    tokens, context_info = extract_tokens_with_context(
        corpus,
        config,
        window_size=context_window_size,
        min_freq=context_min_freq,
        show_progress=show_progress,
    )

    # Limit to top_n tokens if specified
    if top_n is not None and top_n < len(tokens):
        sorted_tokens = sorted(
            [(t, f) for t, f in tokens.items()], key=lambda x: x[1], reverse=True
        )[:top_n]

        tokens = {t: f for t, f in sorted_tokens}

        if extract_context:
            context_info = {t: c for t, c in context_info.items() if t in tokens}

    return tokens, context_info
