"""
Context building module for chord optimization.

This module analyzes segmentation output to track direct adjacency 
relationships between selected tokens.
"""

import logging
from collections import defaultdict
from typing import Dict, List

from src.common.shared_types import TokenData
from src.token_generation.text_segmentation import TextSegment

logger = logging.getLogger(__name__)


def build_adjacency_context(
    segmentation: List[TextSegment], selected_tokens: List[TokenData]
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Build adjacency context from segmentation output.
    
    Args:
        segmentation: List of text segments from optimal segmentation
        selected_tokens: List of selected tokens to track context for
    
    Returns:
        Dictionary mapping token strings to adjacency data:
        {
            "token1": {
                "preceding": {"token2": count, "token3": count},
                "following": {"token4": count, "token5": count}
            }
        }
    """
    # Create set of selected token strings for fast lookup
    selected_token_strings = {token.lower for token in selected_tokens}
    
    # Initialize adjacency tracking
    adjacency_data = {}
    for token_str in selected_token_strings:
        adjacency_data[token_str] = {
            "preceding": defaultdict(int),
            "following": defaultdict(int)
        }
    
    # Process segmentation to find direct adjacencies
    for i, segment in enumerate(segmentation):
        current_token = segment.token_text
        
        # Only track context for selected tokens
        if current_token not in selected_token_strings:
            continue
            
        # Check preceding token
        if i > 0:
            prev_segment = segmentation[i - 1]
            prev_token = prev_segment.token_text
            if prev_token in selected_token_strings:
                adjacency_data[current_token]["preceding"][prev_token] += 1
        
        # Check following token  
        if i < len(segmentation) - 1:
            next_segment = segmentation[i + 1]
            next_token = next_segment.token_text
            if next_token in selected_token_strings:
                adjacency_data[current_token]["following"][next_token] += 1
    
    # Convert defaultdicts to regular dicts
    result = {}
    for token_str, data in adjacency_data.items():
        result[token_str] = {
            "preceding": dict(data["preceding"]),
            "following": dict(data["following"])
        }
    
    return result


def add_adjacency_context_to_tokens(
    selected_tokens: List[TokenData], segmentation: List[TextSegment]
) -> None:
    """
    Add adjacency context information to selected tokens in-place.
    
    Args:
        selected_tokens: List of selected tokens to update
        segmentation: List of text segments from optimal segmentation
    """
    logger.info("Building adjacency context from segmentation...")
    
    # Build adjacency data
    adjacency_data = build_adjacency_context(segmentation, selected_tokens)
    
    # Update tokens with adjacency information
    tokens_updated = 0
    for token in selected_tokens:
        if token.lower in adjacency_data:
            token.adjacent_tokens = adjacency_data[token.lower]
            tokens_updated += 1
    
    logger.info(f"Added adjacency context to {tokens_updated} tokens")
    
    # Log some statistics
    total_preceding = sum(
        len(data["preceding"]) 
        for data in adjacency_data.values()
    )
    total_following = sum(
        len(data["following"]) 
        for data in adjacency_data.values()
    )
    
    logger.debug(f"Total preceding relationships: {total_preceding}")
    logger.debug(f"Total following relationships: {total_following}")
