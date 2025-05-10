"""
Token scoring module for calculating discomfort and candidate scores.

This module provides functions to calculate:
1. Discomfort scores - representing the typing cost per use of a token
2. Candidate scores - representing the overall value of a token based on frequency and discomfort
"""

import logging
from typing import Dict, List

from src.common.shared_types import TokenData

logger = logging.getLogger(__name__)


def calculate_discomfort_score(
    token: TokenData, layout_comfort: Dict[str, float] = None
) -> float:
    """Calculate the discomfort score for a token (cost per use).

    Args:
        token: The token to score
        layout_comfort: Optional dict mapping characters to comfort scores

    Returns:
        Discomfort score (lower is better)
    """
    # TODO: needs additional argument to get the scores of subtokens (probably
    # just the list of selected_tokens)
    # TODO: Split TokenData score into 2 separate fields: comfort_score and
    # candidate_score!! Otherwise it gets to confusing to keep track

    # Ideally, this would also make use of of the assigned chord comfort

    # For single characters, use layout comfort
    if len(token.lower) == 1:
        if layout_comfort and token.lower in layout_comfort:
            return layout_comfort[token.lower] / 10
        # TODO: else assign layout_comfort[unknown] -> needs to be added to
        # config
    else:
        score = 0
        for subtoken in token.best_current_combination:
            # TODO: Needs to be subtoken.discomfort_score instead
            score += 1
        return score


def calculate_candidate_score(token: TokenData, text_length: int) -> float:
    """Calculate the candidate score for token selection based on frequency and discomfort.

    Args:
        token: The token to score
        text_length: Length of the entire text for normalization

    Returns:
        Candidate score (higher is better)
    """
    # Discomfort score (cost per use) - lower is better
    discomfort = token.score  # We store discomfort in the score field for compatibility

    # Frequency benefit - normalized by text length
    frequency_factor = token.text_count / text_length

    # Candidate score: higher frequency and lower discomfort is better
    # Since lower discomfort is better, we divide by discomfort
    return frequency_factor / max(0.001, discomfort)
