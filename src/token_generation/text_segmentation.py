"""
Text segmentation module with dynamic programming optimization.

This module provides functionality to segment text optimally using
currently selected tokens via dynamic programming.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Tuple

from src.common.shared_types import TokenData

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """Represents a segment of text in the optimal segmentation"""

    token_text: str  # The token text (lowercase)
    start_pos: int  # Start position in the original text
    end_pos: int  # End position in the original text
    token_data: TokenData  # Associated token data


def find_optimal_text_segmentation(
    text: str, selected_tokens: List[TokenData]
) -> List[TextSegment]:
    """Find optimal segmentation that minimizes total usage cost.

    Uses dynamic programming to find the segmentation with minimum sum of token usage costs.

    Args:
        text: Input text to segment (should be lowercase)
        selected_tokens: List of available tokens with their usage costs

    Returns:
        List of TextSegment objects representing the minimum cost segmentation

    Examples:
        >>> # If "here" costs 1.0 but h+e+r+e costs 4.2, "here" will be chosen
        >>> # If "the" costs 0.5 but t+h+e costs 3.2, "the" will be chosen
        >>> tokens = [create_token('h', 1.2), create_token('e', 1.0), ...]
        >>> segmentation = find_optimal_text_segmentation("here", tokens)
        >>> [seg.token_text for seg in segmentation]
        ['here']
    """
    n = len(text)

    # Create token lookup dictionary for O(1) access
    token_dict = {token.lower: token for token in selected_tokens}

    # DP array: dp[i] = (min_cost_to_position_i, best_prev_position)
    # dp[i] represents minimum cost to segment text[0:i]
    dp = [(float("inf"), -1) for _ in range(n + 1)]
    dp[0] = (0.0, -1)  # Base case: empty string has cost 0

    # For each position in the text
    for i in range(1, n + 1):
        # Try all possible tokens ending at position i
        for j in range(i):
            candidate_token = text[j:i].lower()

            # Check if this substring is a valid token
            if candidate_token in token_dict:
                token_data = token_dict[candidate_token]
                cost = token_data.usage_cost

                # Calculate total cost if we use this token
                total_cost = dp[j][0] + cost

                # Update if this gives a better solution
                if total_cost < dp[i][0]:
                    dp[i] = (total_cost, j)

    # Reconstruct the optimal segmentation by backtracking
    result = []
    pos = n

    while pos > 0:
        start_pos = dp[pos][1]
        token_text = text[start_pos:pos].lower()
        token_data = token_dict[token_text]

        # Create TextSegment for this token
        segment = TextSegment(
            token_text=token_text,
            start_pos=start_pos,
            end_pos=pos,
            token_data=token_data,
        )
        result.append(segment)
        pos = start_pos

    # Reverse to get correct order (we built it backwards)
    return result[::-1]


def find_optimal_text_segmentation_in_chunks(
    text: str, selected_tokens: List[TokenData], chunk_size: int = 10000
) -> List[TextSegment]:
    """Find optimal segmentation in chunks for memory efficiency.

    Processes text in overlapping chunks to handle very large texts while
    maintaining optimal segmentation across chunk boundaries.

    Args:
        text: Input text to segment
        selected_tokens: List of available tokens
        chunk_size: Size of each chunk to process

    Returns:
        List of TextSegment objects representing optimal segmentation
    """
    if len(text) <= chunk_size:
        return find_optimal_text_segmentation(text, selected_tokens)

    # Find maximum token length for overlap calculation
    max_token_len = max(len(token.lower) for token in selected_tokens)
    overlap = min(max_token_len * 2, chunk_size // 4)

    result = []
    pos = 0

    while pos < len(text):
        # Calculate chunk boundaries
        chunk_start = pos
        chunk_end = min(pos + chunk_size, len(text))

        # Add overlap for all chunks except the first
        if pos > 0:
            chunk_start = max(0, pos - overlap)

        # Extract chunk
        chunk = text[chunk_start:chunk_end]

        # Find optimal segmentation for this chunk
        chunk_segments = find_optimal_text_segmentation(chunk, selected_tokens)

        # Filter segments that belong to our current position
        for segment in chunk_segments:
            global_start = chunk_start + segment.start_pos
            global_end = chunk_start + segment.end_pos

            # Only include segments that start at or after our current position
            if global_start >= pos:
                adjusted_segment = TextSegment(
                    token_text=segment.token_text,
                    start_pos=global_start,
                    end_pos=global_end,
                    token_data=segment.token_data,
                )
                result.append(adjusted_segment)
                pos = global_end

    return result


def visualize_text_segmentation(
    segmentation: List[TextSegment],
    segment_length: int,
    segments_to_show: int,
) -> str:
    """Visualize segments of the text segmentation using | as separators.

    Args:
        segmentation: List of TextSegment objects
        segment_length: Number of segments to show in each view (not character length)
        segments_to_show: Number of different views to show

    Returns:
        String with visualization showing token text separated by pipes

    Examples:
        >>> segments = [seg1, seg2, seg3, seg4, seg5]  # 5 segments
        >>> visualize_text_segmentation(segments, 3, 2)
        # Shows 2 views, each with up to 3 segments
        >>> # View 1: segments 0-2, View 2: segments 2-4
    """
    if not segmentation:
        return "No segmentation available to visualize"

    total_segments = len(segmentation)
    visualizations = []

    # Calculate segment boundaries for each view
    for view_idx in range(segments_to_show):
        # Calculate starting segment index for this view
        if segments_to_show == 1:
            start_segment = 0
        else:
            # Distribute views evenly across total segments
            start_segment = view_idx * total_segments // segments_to_show

        # Calculate ending segment index (exclusive)
        end_segment = min(start_segment + segment_length, total_segments)

        # Get segments for this view
        view_segments = segmentation[start_segment:end_segment]

        # Build visualization for this view
        view_text = f"Segmentation view {view_idx + 1} [segments {start_segment}:{end_segment}]:\n"

        # Extract token texts and join with pipes
        token_texts = [segment.token_text for segment in view_segments]
        segmented_text = "|".join(token_texts)
        view_text += segmented_text + "\n\n"

        # List the segments with their details
        view_text += "Tokens in this segment:\n"
        for segment in view_segments:
            usage_cost = segment.token_data.usage_cost
            replacement_score = segment.token_data.replacement_score
            token_type = segment.token_data.token_type.name

            view_text += (
                f"[{segment.start_pos}:{segment.end_pos}] '{segment.token_text}' "
                f"(usage_cost: {usage_cost:.4f}, replacement_score: {replacement_score:.6f}, "
                f"type: {token_type})\n"
            )

        visualizations.append(view_text)

    return "\n".join(visualizations)
