"""
Text segmentation module with dynamic programming optimization.

This module provides functionality to segment text optimally using
currently selected tokens via dynamic programming with trie-based search
and persistent caching for improved performance.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.common.shared_types import TokenData

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """Represents a segment of text in the optimal segmentation"""

    token_text: str  # The token text (lowercase)
    start_pos: int  # Start position in the original text
    end_pos: int  # End position in the original text
    token_data: TokenData  # Associated token data


@dataclass
class TrieNode:
    """Node in the token trie structure"""

    children: Dict[str, "TrieNode"]
    is_end_token: bool
    token_data: Optional[TokenData]

    def __init__(self):
        self.children = {}
        self.is_end_token = False
        self.token_data = None


class TokenTrie:
    """Trie data structure for efficient token lookup"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_data: TokenData):
        """Insert a token into the trie"""
        node = self.root
        for char in token_data.lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_token = True
        node.token_data = token_data


@dataclass
class CacheEntry:
    """Cache entry for segmentation results"""

    min_cost: float
    parent_pos: int
    last_token_set_hash: int  # Hash of token set when this was computed

    def __init__(self, min_cost: float, parent_pos: int, token_set_hash: int):
        self.min_cost = min_cost
        self.parent_pos = parent_pos
        self.last_token_set_hash = token_set_hash


def find_optimal_text_segmentation(
    text: str, selected_tokens: List[TokenData]
) -> List[TextSegment]:
    """Find optimal segmentation that minimizes total usage cost.

    Uses dynamic programming with trie-based search and persistent caching for
    improved performance in iterative selection processes. Achieves O(n*L) complexity
    where n is text length and L is maximum token length.

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
    # Build token trie for efficient lookups
    token_trie = build_token_trie(selected_tokens)

    # Create a dictionary for token data lookup
    token_dict = {token.lower: token for token in selected_tokens}

    # Find the maximum token length for optimization
    max_token_length = max(len(token.lower) for token in selected_tokens)

    # Calculate hash of current token set for cache validation
    token_set_hash = hash(frozenset(token.lower for token in selected_tokens))

    # Initialize the search with caching
    n = len(text)
    segmentation = find_optimal_segmentation_with_cache(
        text, token_trie, token_dict, max_token_length, token_set_hash, n
    )

    # Update usage counts for selected tokens
    update_token_usage_counts(segmentation, selected_tokens)

    return segmentation


def build_token_trie(tokens: List[TokenData]) -> TokenTrie:
    """Build a trie from the provided tokens for efficient lookup.

    Args:
        tokens: List of tokens to build the trie from

    Returns:
        TokenTrie with all tokens inserted
    """
    trie = TokenTrie()
    for token in tokens:
        trie.insert(token)
    return trie


def find_optimal_segmentation_with_cache(
    text: str,
    token_trie: TokenTrie,
    token_dict: Dict[str, TokenData],
    max_token_length: int,
    token_set_hash: int,
    text_length: int,
) -> List[TextSegment]:
    """Find the optimal segmentation using dynamic programming with caching.

    Args:
        text: Input text to segment
        token_trie: Trie containing all tokens
        token_dict: Dictionary for O(1) token data lookup
        max_token_length: Maximum token length for optimization
        token_set_hash: Hash of the current token set for cache validation
        text_length: Length of the input text

    Returns:
        List of TextSegment objects representing the optimal segmentation
    """
    # Static cache to persist between calls
    if not hasattr(find_optimal_segmentation_with_cache, "position_cache"):
        find_optimal_segmentation_with_cache.position_cache = {}

    # DP array: dp[i] = (min_cost_to_position_i, best_prev_position)
    dp = [(float("inf"), -1) for _ in range(text_length + 1)]
    dp[0] = (0.0, -1)  # Base case: empty string has cost 0

    # Fill the DP table with caching
    for i in range(1, text_length + 1):
        # Check if we have a cached result for this position with the current token set
        cache_key = (i, token_set_hash)
        if cache_key in find_optimal_segmentation_with_cache.position_cache:
            cache_entry = find_optimal_segmentation_with_cache.position_cache[cache_key]
            dp[i] = (cache_entry.min_cost, cache_entry.parent_pos)
            continue

        # Try tokens ending at position i with limited length search
        optimal_segmentation_dp_step(text, i, max_token_length, token_dict, dp)

        # Cache the result
        find_optimal_segmentation_with_cache.position_cache[cache_key] = CacheEntry(
            dp[i][0], dp[i][1], token_set_hash
        )

    # Reconstruct the segmentation by backtracking
    return reconstruct_segmentation(text, dp, token_dict, text_length)


def optimal_segmentation_dp_step(
    text: str,
    pos: int,
    max_token_length: int,
    token_dict: Dict[str, TokenData],
    dp: List[tuple[float, int]],
) -> None:
    """Execute a single DP step for the optimal segmentation algorithm.

    Args:
        text: Input text to segment
        pos: Current position in the text
        max_token_length: Maximum token length to consider
        token_dict: Dictionary for token data lookup
        dp: Dynamic programming array to update
    """
    # Only look back up to max_token_length positions for efficiency
    start_pos = max(0, pos - max_token_length)

    for j in range(start_pos, pos):
        # Skip unreachable positions
        if dp[j][0] == float("inf"):
            continue

        # Check if text[j:i] is a valid token
        candidate_token = text[j:pos].lower()
        if candidate_token in token_dict:
            token_data = token_dict[candidate_token]
            cost = token_data.usage_cost

            # Calculate total cost if we use this token
            total_cost = dp[j][0] + cost

            # Update if this gives a better solution
            if total_cost < dp[pos][0]:
                dp[pos] = (total_cost, j)


def reconstruct_segmentation(
    text: str,
    dp: List[tuple[float, int]],
    token_dict: Dict[str, TokenData],
    text_length: int,
) -> List[TextSegment]:
    """Reconstruct the segmentation by backtracking through the DP array.

    Args:
        text: Input text
        dp: Dynamic programming array
        token_dict: Dictionary for token data lookup
        text_length: Length of the input text

    Returns:
        List of TextSegment objects in the correct order
    """
    result = []
    pos = text_length

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


def update_token_usage_counts(
    segmentation: List[TextSegment], tokens: List[TokenData]
) -> None:
    """Update usage counts for tokens based on the segmentation.

    Args:
        segmentation: List of TextSegment objects
        tokens: List of TokenData objects to update
    """
    # Reset usage counts
    for token in tokens:
        token.usage_count = 0

    # Count usage in the segmentation
    for segment in segmentation:
        token_text = segment.token_text
        for token in tokens:
            if token.lower == token_text:
                token.usage_count += 1
                break


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
                f"(replacement_score: {replacement_score:.6f}, "
                f"usage_cost: {usage_cost:.4f}, "
                f"type: {token_type})\n"
            )

        visualizations.append(view_text)

    return "\n".join(visualizations)
