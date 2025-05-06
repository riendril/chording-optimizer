"""
Context building module for chord optimization.

This module focuses on building context relationships between tokens,
analyzing their co-occurrence patterns and substring relationships.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from src.common.config import GeneratorConfig
from src.common.shared_types import ContextInfo, TokenCollection, TokenData

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
# Public API
# -----------------


def add_context_to_token_collection(
    collection: TokenCollection,
    corpus_text: str,
    window_size: int = 2,
    min_freq: int = 2,
    show_progress: bool = True,
) -> TokenCollection:
    """
    Add context information to an existing token collection.

    Args:
        collection: Existing token collection
        corpus_text: The corpus text to extract context from
        window_size: Size of window for context relationships
        min_freq: Minimum frequency for context relationships
        show_progress: Whether to show progress updates

    Returns:
        Updated token collection with context information
    """
    if show_progress:
        print("Preprocessing text for context building...")

    processed_text = preprocess_text(corpus_text)

    # Get token frequencies as dictionary
    token_freqs = {token.original: token.frequency for token in collection.tokens}

    if show_progress:
        print("Building context information...")

    # Extract context information
    context_info = build_context_information(
        processed_text, token_freqs, window_size, min_freq
    )

    if show_progress:
        print(f"Built context information for {len(context_info)} tokens")

    # Create a new collection with the same tokens but added context
    token_map = {token.original: token for token in collection.tokens}
    updated_tokens = []

    for token_data in collection.tokens:
        new_token = TokenData.from_token(
            token=token_data.original,
            frequency=token_data.frequency,
            rank=token_data.rank,
            zipf_weight=token_data.zipf_weight,
            score=token_data.score,
            difficulty=(
                token_data.difficulty if hasattr(token_data, "difficulty") else 0.0
            ),
        )

        # Add context if available
        if token_data.original in context_info:
            new_token.context = context_info[token_data.original]

        updated_tokens.append(new_token)

    # Create and return the new collection
    return TokenCollection(
        name=f"{collection.name}_with_context",
        tokens=updated_tokens,
        ordered_by_frequency=collection.ordered_by_frequency,
        source=collection.source,
    )


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


def extract_tokens_with_context(
    corpus: str,
    config: GeneratorConfig,
    window_size: int = 2,
    min_freq: int = 2,
    show_progress: bool = True,
) -> Tuple[Dict[str, int], Dict[str, ContextInfo]]:
    """
    Extract tokens and build context information in a single pass.

    This function provides backward compatibility with the original API.

    Args:
        corpus: Raw input text
        config: Generator configuration
        window_size: Size of window for context relationships
        min_freq: Minimum frequency for context relationships
        show_progress: Whether to show progress updates

    Returns:
        Tuple of (token frequencies, context information)
    """
    from src.token_generation.token_extractor import extract_tokens_from_text

    tokens = extract_tokens_from_text(corpus, config, show_progress)

    if show_progress:
        print("Building context information...")

    processed_text = preprocess_text(corpus)
    context_info = build_context_information(
        processed_text, tokens, window_size, min_freq
    )

    if show_progress:
        print(f"Built context information for {len(context_info)} tokens.")

    return tokens, context_info


def add_context_to_file(
    token_collection_path: str,
    corpus_path: str,
    output_path: Optional[str] = None,
    window_size: int = 2,
    min_freq: int = 2,
    show_progress: bool = True,
) -> TokenCollection:
    """
    Add context information to a token collection file.

    Args:
        token_collection_path: Path to token collection file
        corpus_path: Path to corpus file
        output_path: Path to save the updated collection (defaults to original path)
        window_size: Size of window for context relationships
        min_freq: Minimum frequency for context relationships
        show_progress: Whether to show progress updates

    Returns:
        Updated token collection with context information
    """
    from src.token_generation.token_extractor import read_corpus_from_file

    # Load token collection
    collection = TokenCollection.load_from_file(token_collection_path)

    # Load corpus
    corpus = read_corpus_from_file(corpus_path)

    # Add context
    updated_collection = add_context_to_token_collection(
        collection, corpus, window_size, min_freq, show_progress
    )

    # Save to file if output path is specified
    if output_path:
        updated_collection.save_to_file(output_path)
        if show_progress:
            print(f"Saved updated collection to {output_path}")

    return updated_collection
