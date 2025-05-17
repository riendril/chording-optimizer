"""
Token scoring module for chord optimization.

This module provides functions to calculate:
1. Usage cost - representing the typing cost per use of a token (static)
2. Replacement score - overall value to replace a token with a chord based on
frequency and usage cost (dynamic)

The usage cost is cached to avoid recalculation for tokens with the same composition.
"""

import logging
from functools import lru_cache
from typing import Dict, List, Tuple

from src.common.shared_types import TokenData

logger = logging.getLogger(__name__)

# Global cache for usage costs based on token composition
_usage_cost_cache = {}


def calculate_usage_cost(
    token: TokenData,
    selected_tokens: List[TokenData],
    layout_usage_cost: Dict[str, float],
) -> float:
    """Calculate the usage cost for a token (cost per use).

    The usage cost is calculated once for each unique token composition
    and cached for efficiency.

    Args:
        token: The token to score
        selected_tokens: List of currently selected tokens (for subtoken costs)
        layout_usage_cost: Dict mapping characters to usage costs (required)

    Returns:
        Usage cost (lower is better)

    Examples:
        >>> # Single character with layout usage cost
        >>> token = TokenData(lower='a', length=1, token_type=TokenType.SINGLE_CHARACTER,
        ...                   text_count=100, usage_count=100, rank=1,
        ...                   usage_cost=0.0, replacement_score=0.0, selected=True,
        ...                   best_current_combination=['a'])
        >>> layout_usage_cost = {'a': 1.0, 'b': 2.0}
        >>> calculate_usage_cost(token, None, layout_usage_cost)
        1.0

        >>> # Multi-character token with subtokens
        >>> token2 = TokenData(lower='abc', length=3, token_type=TokenType.NGRAM_LETTERS_ONLY,
        ...                    text_count=50, usage_count=50, rank=2,
        ...                    usage_cost=0.0, replacement_score=0.0, selected=False,
        ...                    best_current_combination=['a', 'b', 'c'])
        >>> a_token = TokenData(lower='a', length=1, token_type=TokenType.SINGLE_CHARACTER,
        ...                     text_count=100, usage_count=100, rank=1,
        ...                     usage_cost=1.0, replacement_score=0.0, selected=True,
        ...                     best_current_combination=['a'])
        >>> selected_tokens = [a_token]
        >>> calculate_usage_cost(token2, selected_tokens, layout_usage_cost)
        3.0
    """

    # TODO: Ideally, this would also make use of of the assigned chord comfort

    # Create a cache key from the token's best_current_combination
    cache_key = tuple(token.best_current_combination)

    # Check if already cached
    if cache_key in _usage_cost_cache:
        return _usage_cost_cache[cache_key]

    # For single characters, use layout usage cost
    if len(token.lower) == 1:
        if token.lower in layout_usage_cost:
            cost = layout_usage_cost[token.lower]
        else:
            # Use the default cost for unknown characters
            cost = layout_usage_cost.get("unknown", 15)
            logger.debug(f"Using default cost for unknown character: '{token.lower}'")
    else:
        # For multi-character tokens, sum the costs of subtokens
        cost = 0

        # Create dictionary of selected tokens for faster lookup
        selected_dict = {}
        if selected_tokens:
            selected_dict = {t.lower: t for t in selected_tokens}

        for subtoken in token.best_current_combination:
            # Use the cached usage_cost of the selected subtoken
            subtoken_cost = selected_dict[subtoken].usage_cost
            cost += subtoken_cost

    # Cache the result
    _usage_cost_cache[cache_key] = cost
    return cost


def calculate_replacement_score(
    token: TokenData,
    text_length: int,
    selected_tokens: List[TokenData],
    layout_usage_cost: Dict[str, float],
) -> float:
    """Calculate the replacement score for a token based on frequency in current
    segmentation and usage cost.

    The replacement score represents the overall value of adding a token to the
    selected set, replacing it with a chord. Higher scores indicate more
    valuable tokens.

    Args:
        token: The token to score
        text_length: Length of the entire text for normalization
        selected_tokens: List of currently selected tokens (for usage cost calculation)
        layout_usage_cost: Dict mapping characters to usage costs (required)

    Returns:
        Replacement score (higher is better)

    Examples:
        >>> token = TokenData(lower='the', length=3, token_type=TokenType.FULL_WORD,
        ...                   text_count=1000, usage_count=1000, rank=1,
        ...                   usage_cost=3.0, replacement_score=0.0, selected=False,
        ...                   best_current_combination=['t', 'h', 'e'])
        >>> calculate_replacement_score(token, 100000, [], {})
        0.00003
    """
    # Frequency benefit - normalized by text length
    frequency_factor = token.usage_count / text_length

    # Replacement score: How valuable is the replacement of this token
    return frequency_factor * token.usage_cost


def update_token_scores_and_sort(
    tokens: List[TokenData],
    text_length: int,
    selected_tokens: List[TokenData],
    layout_usage_cost: Dict[str, float],
) -> None:
    """Update both usage costs and replacement scores for a list of tokens in place.
    Args:
        tokens: List of tokens to update scores for
        text_length: Length of the entire text for normalization
        selected_tokens: List of currently selected tokens (for usage cost calculation)
        layout_usage_cost: Dict mapping characters to usage costs (required)
    """
    for token in tokens:
        # Calculate usage cost (static score)
        token.usage_cost = calculate_usage_cost(
            token, selected_tokens, layout_usage_cost
        )
        # Calculate replacement score (dynamic value)
        token.replacement_score = calculate_replacement_score(
            token, text_length, selected_tokens, layout_usage_cost
        )

    # Sort by replacement score and assign ranks
    tokens.sort(key=lambda t: t.replacement_score, reverse=True)

    for i, token in enumerate(tokens):
        token.rank = i + 1


def get_cache_stats():
    """Get statistics about the usage cost cache.

    Returns:
        Dict containing cache size and other statistics
    """
    return {
        "cache_size": len(_usage_cost_cache),
        "cache_memory_estimate": len(_usage_cost_cache) * 24,  # Rough estimate in bytes
    }
