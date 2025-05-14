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
    """Find optimal segmentation of text using currently selected tokens with
    dynamic programming.

    Args:
        text: Input text to tokenize
        selected_tokens: List of currently selected tokens

    Returns:
        List of TextSegment objects representing the optimal segmentation
    """
    # Create a dict of selected token strings to TokenData for fast lookup
    token_dict = {token.lower: token for token in selected_tokens}
    # TODO: Is that actually useful?

    # DP array: best[i] = best segmentation up to position i
    # Each entry contains (score, last_token_start)
    n = len(text)
    best = [(0, 0) for _ in range(n + 1)]

    # Bottom-up DP
    for i in range(1, n + 1):
        # Default: use previous best + single character
        # For single characters, use their usage cost from the token
        char = text[i - 1 : i].lower()
        char_token = token_dict.get(char)

        # Use inverse of usage cost for the DP score (higher = better segmentation)
        # TODO: What exactly do these numbers mean? Looks weirdly hard coded
        best[i] = (best[i - 1][0] + 1.0 / char_token.usage_cost, i - 1)

        # Try all possible last tokens ending at position i
        max_token_len = min(i, max(len(t.lower) for t in selected_tokens))

        for length in range(1, max_token_len + 1):
            start = i - length
            candidate = text[start:i].lower()

            if candidate in token_dict:
                # Use usage cost (lower is better for segmentation)
                token_usage_cost = token_dict[candidate].usage_cost

                # For segmentation, we want lower usage cost = better score
                # So we use inverse of usage cost (higher = better)
                token_value = 1.0 / max(0.1, token_usage_cost)

                candidate_score = best[start][0] + token_value
                if candidate_score > best[i][0]:
                    best[i] = (candidate_score, start)

    # Reconstruct the tokenization
    result = []
    pos = n

    while pos > 0:
        start = best[pos][1]
        token_text = text[start:pos].lower()

        if token_text in token_dict:
            token_data = token_dict[token_text]
        else:
            # Single character case
            char = text[start:pos].lower()
            # Find the single character token
            token_data = next((t for t in selected_tokens if t.lower == char), None)

        result.append(
            TextSegment(
                token_text=token_text,
                start_pos=start,
                end_pos=pos,
                token_data=token_data,
            )
        )
        pos = start

    # Reverse to get tokens in order
    return result[::-1]


def find_optimal_text_segmentation_in_chunks(
    text: str, selected_tokens: List[TokenData], chunk_size: int = 10000
) -> List[TextSegment]:
    """Find optimal segmentation of text in chunks for better memory efficiency with large corpuses.

    Args:
        text: Input text to tokenize
        selected_tokens: List of currently selected tokens
        chunk_size: Size of chunks to process

    Returns:
        List of TextSegment objects
    """
    result = []
    n = len(text)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        # Find optimal segmentation for this chunk
        chunk_segments = find_optimal_text_segmentation(chunk, selected_tokens)

        # Adjust positions to global coordinates
        adjusted_segments = [
            TextSegment(
                token_text=segment.token_text,
                start_pos=start + segment.start_pos,
                end_pos=start + segment.end_pos,
                token_data=segment.token_data,
            )
            for segment in chunk_segments
        ]

        result.extend(adjusted_segments)

    return result


def visualize_text_segmentation(
    segmentation: List[TextSegment],
    segment_length: int,
    segments_to_show: int,
) -> str:
    """Visualize segments of the text segmentation using | as separators.

    Args:
        segmentation: List of TextSegment objects
        segment_length: Length of text segment to visualize
        segments_to_show: Number of segments to visualize

    Returns:
        String with visualization
    """
    if not segmentation:
        return "No segmentation available to visualize"

    visualizations = []

    # Reconstruct the full text from segmentation to calculate total length
    full_text = ""
    for segment in segmentation:
        full_text += segment.token_text

    # For each visualization requested
    for i in range(segments_to_show):
        # Calculate start position for this visualization
        if segments_to_show == 1:
            # Show from beginning if only one segment requested
            start_pos = 0
        else:
            # Distribute segments evenly across the text
            start_pos = i * len(full_text) // segments_to_show

        # Find the ending position
        end_pos = start_pos + segment_length

        # Find segments that fall within our visualization window
        relevant_segments = []
        current_pos = 0

        for segment in segmentation:
            segment_start = current_pos
            segment_end = current_pos + len(segment.token_text)

            # Check if this segment overlaps with our visualization window
            if segment_start < end_pos and segment_end > start_pos:
                relevant_segments.append((segment, segment_start, segment_end))

            current_pos = segment_end

            # Stop if we've passed our window
            if segment_start >= end_pos:
                break

        # Build the visualization
        visualization = f"Segmentation view {i+1} [position {start_pos}:{end_pos}]:\n"

        # Create the text line for this window
        text_line = ""
        marker_line = ""

        # Start with the first character of our window
        current_char_pos = start_pos

        for segment, segment_start, segment_end in relevant_segments:
            # Add text from this segment that falls in our window
            text_start_in_segment = max(0, start_pos - segment_start)
            text_end_in_segment = min(len(segment.token_text), end_pos - segment_start)

            if text_end_in_segment > text_start_in_segment:
                segment_text = segment.token_text[
                    text_start_in_segment:text_end_in_segment
                ]
                text_line += segment_text

                # Add markers (| at the beginning of each token)
                if current_char_pos == max(start_pos, segment_start):
                    marker_line += "|"

                # Add spaces under the text for proper alignment
                marker_line += " " * (
                    len(segment_text)
                    - (1 if current_char_pos == max(start_pos, segment_start) else 0)
                )

                current_char_pos += len(segment_text)

        # Add final marker if we ended exactly at a token boundary
        if current_char_pos < end_pos and len(text_line) < segment_length:
            marker_line += "|"

        visualization += text_line + "\n"
        visualization += marker_line + "\n\n"

        # Add token details
        visualization += "Tokens in this segment:\n"
        for segment, segment_start, segment_end in relevant_segments:
            # Calculate positions relative to our visualization window
            rel_start = max(0, segment_start - start_pos)
            rel_end = min(segment_length, segment_end - start_pos)

            usage_cost = segment.token_data.usage_cost
            replacement_score = segment.token_data.replacement_score
            token_type = segment.token_data.token_type.name

            visualization += (
                f"[{rel_start}:{rel_end}] '{segment.token_text}' "
                f"(usage_cost: {usage_cost:.4f}, replacement_score: {replacement_score:.6f}, "
                f"type: {token_type})\n"
            )

        visualizations.append(visualization)

    return "\n".join(visualizations)
