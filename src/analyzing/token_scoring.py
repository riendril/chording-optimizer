"""
Token scoring module for chord optimization.

This module calculates scores for tokens based on typing difficulty,
token frequency, length, and contextual relationships.
"""

import math
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.common.config import GeneratorConfig
from src.common.shared_types import ContextInfo, Finger, TokenCollection, TokenData

# -----------------
# Core Difficulty Functions
# -----------------


@lru_cache(maxsize=128)
def calculate_key_difficulty(char: str, layout_config: Dict) -> float:
    """
    Calculate difficulty value for a single key.

    Args:
        char: Character to evaluate
        layout_config: Keyboard layout configuration

    Returns:
        Difficulty score (higher is more difficult)
    """
    comfort_values = layout_config.get("comfort", {})
    return comfort_values.get(char, 10.0)  # Default to 10 for unknown keys


def get_finger_for_key(char: str, layout_config: Dict) -> Optional[Finger]:
    """
    Get the finger used for a character.

    Args:
        char: Character to evaluate
        layout_config: Keyboard layout configuration

    Returns:
        Finger enum or None if not found
    """
    finger_map = layout_config.get("fingers", {})
    finger_name = finger_map.get(char)
    return Finger[finger_name] if finger_name else None


def check_same_finger(a: str, b: str, layout_config: Dict) -> bool:
    """
    Check if two characters are typed with the same finger.

    Args:
        a: First character
        b: Second character
        layout_config: Keyboard layout configuration

    Returns:
        True if both characters use the same finger
    """
    finger_a = get_finger_for_key(a, layout_config)
    finger_b = get_finger_for_key(b, layout_config)
    return finger_a is not None and finger_b is not None and finger_a == finger_b


def calculate_key_distance(a: str, b: str, layout_config: Dict) -> float:
    """
    Calculate physical distance between two keys.

    Args:
        a: First character
        b: Second character
        layout_config: Keyboard layout configuration

    Returns:
        Euclidean distance between keys
    """
    positions = layout_config.get("positions", {})

    if a not in positions or b not in positions:
        return 0.0

    a_pos, b_pos = positions[a], positions[b]
    return math.sqrt((a_pos[0] - b_pos[0]) ** 2 + (a_pos[1] - b_pos[1]) ** 2)


def calculate_base_difficulty(token: str, layout_config: Dict) -> float:
    """
    Calculate cumulative base difficulty of all keys in token.

    Args:
        token: Token to evaluate
        layout_config: Keyboard layout configuration

    Returns:
        Total base difficulty score
    """
    return sum(calculate_key_difficulty(char, layout_config) for char in token)


def calculate_transition_difficulty(
    token: str, layout_config: Dict, prev_token: str = "", next_token: str = ""
) -> float:
    """
    Calculate difficulty of transitions between adjacent keys.

    Args:
        token: Token to evaluate
        layout_config: Keyboard layout configuration
        prev_token: Previous token (for context-aware scoring)
        next_token: Next token (for context-aware scoring)

    Returns:
        Transition difficulty score
    """
    if not token:
        return 0.0

    difficulty = 0.0

    # Check transition from previous token's last character to this token's first
    if prev_token and token:
        prev_last = prev_token[-1]
        curr_first = token[0]

        if check_same_finger(prev_last, curr_first, layout_config):
            difficulty += calculate_key_distance(prev_last, curr_first, layout_config)

    # Check transitions within the current token
    for i in range(len(token) - 1):
        a, b = token[i], token[i + 1]
        if check_same_finger(a, b, layout_config):
            difficulty += calculate_key_distance(a, b, layout_config)

    # Check transition from this token's last character to next token's first
    if token and next_token:
        curr_last = token[-1]
        next_first = next_token[0]

        if check_same_finger(curr_last, next_first, layout_config):
            difficulty += calculate_key_distance(curr_last, next_first, layout_config)

    return difficulty


def calculate_typing_difficulty(
    token: str, layout_config: Dict, prev_token: str = "", next_token: str = ""
) -> float:
    """
    Calculate overall typing difficulty score.

    Args:
        token: Token to evaluate
        layout_config: Keyboard layout configuration
        prev_token: Previous token (for context-aware scoring)
        next_token: Next token (for context-aware scoring)

    Returns:
        Total typing difficulty
    """
    base_diff = calculate_base_difficulty(token, layout_config)
    trans_diff = calculate_transition_difficulty(
        token, layout_config, prev_token, next_token
    )
    return base_diff + trans_diff


# -----------------
# Token Scoring Functions
# -----------------


def calculate_length_benefit(length: int, config: GeneratorConfig) -> float:
    """
    Calculate benefit from token length (longer tokens save more keystrokes).

    Args:
        length: Token length
        config: Generator configuration

    Returns:
        Length benefit factor
    """
    length_exponent = config.token_analysis.get("length_benefit_exponent", 1.5)
    return length**length_exponent


@lru_cache(maxsize=1024)
def apply_zipf_weight(rank: int, total_tokens: int) -> float:
    """
    Apply Zipf's law weighting to score based on token rank.

    Args:
        rank: Token rank by frequency (0-based)
        total_tokens: Total number of tokens

    Returns:
        Zipf weight factor
    """
    # Calculate harmonic number for normalization
    harmonic_n = sum(1.0 / i for i in range(1, total_tokens + 1))

    # Apply Zipf's law (rank + 1 to avoid division by zero)
    return 1.0 / ((rank + 1) * harmonic_n)


def calculate_initial_score(
    token: str,
    frequency: int,
    rank: int,
    total_tokens: int,
    difficulty: float,
    config: GeneratorConfig,
) -> float:
    """
    Calculate initial token score based on frequency, length, and difficulty.

    Args:
        token: Token to score
        frequency: Token frequency in corpus
        rank: Token rank by frequency (0-based)
        total_tokens: Total number of tokens
        difficulty: Typing difficulty score
        config: Generator configuration

    Returns:
        Token score (higher means more valuable for chording)
    """
    # Calculate component factors
    zipf_weight = apply_zipf_weight(rank, total_tokens)
    length_benefit = calculate_length_benefit(len(token), config)

    # Difficulty factor (add 1 to ensure positive value)
    difficulty_factor = difficulty + 1.0

    # Final score: higher makes token more attractive for chording
    # Frequency * length benefit * difficulty = value of assigning chord
    return frequency * length_benefit * difficulty_factor * zipf_weight


def calculate_context_adjustment(
    token: TokenData, selected_tokens: Set[str], context_weight: float = 0.2
) -> float:
    """
    Calculate context-based score adjustment.

    Args:
        token: Token data with context information
        selected_tokens: Set of already selected tokens
        context_weight: Weight factor for context adjustments

    Returns:
        Context adjustment factor (multiplier)
    """
    # Start with no adjustment
    adjustment = 0.0

    # Skip if no context information
    if not hasattr(token, "context") or not token.context:
        return 1.0  # No adjustment

    context = token.context

    # Check substring relationships
    for selected in selected_tokens:
        # If this token contains a selected token as substring, devalue it
        if selected in token.original and len(selected) < len(token.original):
            adjustment -= 0.3 * token.score

        # If this token is contained in a selected token, devalue it
        if token.original in selected and len(token.original) < len(selected):
            adjustment -= 0.5 * token.score

    # Check preceding tokens - if frequently preceded by selected tokens, increase value
    for prev_token, freq in context.preceding.items():
        if prev_token in selected_tokens:
            adjustment += 0.1 * freq * token.score

    # Check following tokens - if frequently followed by selected tokens, increase value
    for next_token, freq in context.following.items():
        if next_token in selected_tokens:
            adjustment += 0.1 * freq * token.score

    # Apply context weight and convert to multiplier
    # Ensure adjustment doesn't completely eliminate score
    return max(1.0 - (context_weight * adjustment / token.score), 0.1)


# -----------------
# Public Interface Functions
# -----------------


def score_token_collection(
    token_collection: TokenCollection, layout_config: Dict, config: GeneratorConfig
) -> TokenCollection:
    """
    Score all tokens in a collection.

    Args:
        token_collection: Collection of tokens to score
        layout_config: Keyboard layout configuration
        config: Generator configuration

    Returns:
        Updated token collection with scores
    """
    total_tokens = len(token_collection.tokens)

    # Create a new list to store updated tokens
    updated_tokens = []

    for token_data in token_collection.tokens:
        # Calculate typing difficulty
        difficulty = calculate_typing_difficulty(token_data.original, layout_config)

        # Calculate initial score
        score = calculate_initial_score(
            token_data.original,
            token_data.frequency,
            token_data.rank,
            total_tokens,
            difficulty,
            config,
        )

        # Update token with calculated values
        token_data.difficulty = difficulty
        token_data.score = score
        updated_tokens.append(token_data)

    # Create a new collection with updated scores
    return TokenCollection(
        name=token_collection.name,
        tokens=updated_tokens,
        ordered_by_frequency=token_collection.ordered_by_frequency,
        source=token_collection.source,
    )


def recalculate_token_scores(
    token_collection: TokenCollection,
    selected_tokens: Set[str],
    context_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Recalculate scores for tokens based on already selected tokens and their context.

    Args:
        token_collection: Collection of tokens with context info
        selected_tokens: Set of tokens already selected for chords
        context_weight: Weight for context influence on score (0.0-1.0)

    Returns:
        Dictionary mapping tokens to their recalculated scores
    """
    new_scores = {}

    # Create map from token string to TokenData
    token_map = {token.original: token for token in token_collection.tokens}

    # For each token in the collection
    for token in token_collection.tokens:
        # Skip already selected tokens
        if token.original in selected_tokens:
            continue

        # Start with the original score
        base_score = token.score

        # Apply context adjustment
        context_adjustment = calculate_context_adjustment(
            token, selected_tokens, context_weight
        )

        # Calculate adjusted score
        new_scores[token.original] = base_score * context_adjustment

    return new_scores


def get_token_score_function(
    layout_config: Dict, config: GeneratorConfig
) -> Callable[[str, int, int, int], float]:
    """
    Get a scoring function for token selection algorithms.

    Args:
        layout_config: Keyboard layout configuration
        config: Generator configuration

    Returns:
        Function that scores tokens
    """

    def score_function(
        token: str, frequency: int, rank: int, total_tokens: int
    ) -> float:
        difficulty = calculate_typing_difficulty(token, layout_config)
        return calculate_initial_score(
            token, frequency, rank, total_tokens, difficulty, config
        )

    return score_function


def get_score_recalculation_function(
    token_collection: TokenCollection, context_weight: float = 0.2
) -> Callable[[Set[str]], Dict[str, float]]:
    """
    Get a score recalculation function for selection algorithms.

    Args:
        token_collection: Collection of tokens with context info
        context_weight: Weight for context influence

    Returns:
        Function that recalculates scores based on selected tokens
    """

    def recalculation_function(selected_tokens: Set[str]) -> Dict[str, float]:
        return recalculate_token_scores(
            token_collection, selected_tokens, context_weight
        )

    return recalculation_function


# -----------------
# Combined Interface
# -----------------


def prepare_token_scoring(
    token_collection: TokenCollection, layout_config: Dict, config: GeneratorConfig
) -> Tuple[TokenCollection, Callable[[Set[str]], Dict[str, float]]]:
    """
    Prepare token collection with scores and return recalculation function.

    This is the main entry point for the scoring module, providing everything
    needed for selection algorithms in one call.

    Args:
        token_collection: Collection of tokens to score
        layout_config: Keyboard layout configuration
        config: Generator configuration

    Returns:
        Tuple of (scored token collection, score recalculation function)
    """
    # Score all tokens
    scored_collection = score_token_collection(token_collection, layout_config, config)

    # Create recalculation function
    recalculation_function = get_score_recalculation_function(
        scored_collection,
        context_weight=config.chord_assignment.get("context_weight", 0.2),
    )

    return scored_collection, recalculation_function
