"""
Text segmentation module with dynamic programming optimization.

This module provides functionality to segment text optimally using
currently selected tokens via dynamic programming.
"""

import logging
import random
from typing import List, Tuple

from src.common.shared_types import TokenData

logger = logging.getLogger(__name__)


def find_optimal_text_segmentation(
    text: str, selected_tokens: List[TokenData]
) -> List[Tuple[str, int, int, TokenData]]:
    """Find optimal segmentation of text using currently selected tokens with
    dynamic programming.

    Args:
        text: Input text to tokenize
        selected_tokens: List of currently selected tokens

    Returns:
        List of tuples (token_text, start_pos, end_pos, token_data)
        representing the optimal segmentation
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
        # For single characters, use their discomfort score from the token
        char = text[i - 1 : i].lower()
        char_token = token_dict.get(char)

        # Use inverse of discomfort for the DP score (higher = better segmentation)
        # TODO: What exactly do these numbers mean? Looks weirdly hard coded
        best[i] = (best[i - 1][0] + 1.0 / char_token.score, i - 1)

        # Try all possible last tokens ending at position i
        max_token_len = min(i, max(len(t.lower) for t in selected_tokens))

        for length in range(1, max_token_len + 1):
            start = i - length
            candidate = text[start:i].lower()

            if candidate in token_dict:
                # Use discomfort score (lower is better for segmentation)
                token_discomfort = token_dict[candidate].score

                # For segmentation, we want lower discomfort = better score
                # So we use inverse of discomfort (higher = better)
                token_value = 1.0 / max(0.1, token_discomfort)

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

        result.append((token_text, start, pos, token_data))
        pos = start

    # Reverse to get tokens in order
    return result[::-1]


def find_optimal_text_segmentation_in_chunks(
    text: str, selected_tokens: List[TokenData], chunk_size: int = 10000
) -> List[Tuple[str, int, int, TokenData]]:
    """Find optimal segmentation of text in chunks for better memory efficiency with large corpuses.

    Args:
        text: Input text to tokenize
        selected_tokens: List of currently selected tokens
        chunk_size: Size of chunks to process

    Returns:
        List of tuples (token_text, start_pos, end_pos, token_data)
    """
    result = []
    n = len(text)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        # Find optimal segmentation for this chunk
        chunk_tokens = find_optimal_text_segmentation(chunk, selected_tokens)

        # Adjust positions to global coordinates
        adjusted_tokens = [
            (token_text, start + token_start, start + token_end, token_data)
            for token_text, token_start, token_end, token_data in chunk_tokens
        ]

        result.extend(adjusted_tokens)

    return result


def visualize_text_segmentation(
    text: str,
    segmentation: List[Tuple[str, int, int, TokenData]],
    segment_length: int = 100,
    segments_to_show: int = 1,
) -> str:
    """Visualize random segments of the text segmentation using | as separators.

    Args:
        text: Original text
        segmentation: List of tuples (token_text, start_pos, end_pos, token_data)
        segment_length: Length of text segment to visualize
        segments_to_show: Number of random segments to visualize

    Returns:
        String with visualization
    """
    if not segmentation:
        return "No segmentation available to visualize"

    visualizations = []

    # TODO: Why does this function even need the text? The segmentation should
    # be enough, right?
    # FIX: Current output is sth like this: "||||||||||||||||"

    for _ in range(segments_to_show):
        # Find a random starting point
        max_start = max(0, len(text) - segment_length)
        random_start = random.randint(0, max_start)
        random_end = random_start + segment_length

        # Find tokens that overlap with the selected segment
        segment_tokens = []
        for token_text, start_pos, end_pos, token_data in segmentation:
            if start_pos < random_end and end_pos > random_start:
                segment_tokens.append((token_text, start_pos, end_pos, token_data))

        # Create text line with segment
        segment_text = text[random_start:random_end]
        visualization = f"Text segment [{random_start}:{random_end}]:\n{segment_text}\n"

        # Create marker line with | at token boundaries
        markers = ["" for _ in range(segment_length + 1)]

        # Add | at the beginning
        markers[0] = "|"

        # Add | at token boundaries
        for _, start_pos, end_pos, _ in segment_tokens:
            rel_start = start_pos - random_start
            rel_end = end_pos - random_start

            if 0 <= rel_start < segment_length:
                markers[rel_start] = "|"
            if 0 < rel_end <= segment_length:
                markers[rel_end] = "|"

        # Build the marker line
        marker_line = ""
        for i, marker in enumerate(markers):
            if marker:
                marker_line += marker
            else:
                marker_line += " "

        visualization += marker_line + "\n\n"

        # Add token details
        visualization += "Tokens in this segment:\n"
        for token_text, start_pos, end_pos, token_data in segment_tokens:
            rel_start = max(0, start_pos - random_start)
            rel_end = min(segment_length, end_pos - random_start)

            # Calculate candidate score for display
            discomfort = token_data.score
            # FIX: Use candidate_score attribute instead of calculating here
            candidate_score = (
                token_data.text_count / len(text) / discomfort if discomfort > 0 else 0
            )

            visualization += f"[{rel_start}:{rel_end}] '{token_text}' (discomfort: {discomfort:.4f}, candidate_score: {candidate_score:.6f})\n"

        visualizations.append(visualization)

    return "\n".join(visualizations)
