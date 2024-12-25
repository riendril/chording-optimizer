"""
This module generates an optimized set of chords for a set of words following Zipf's law
and multiple weighted optimization constraints.
"""

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, Tuple

# Configuration constants
MAX_CHARS = 5
WEIGHT_FIRST_LAST_CHAR = 0.1  # Reduced to prevent negative costs
WEIGHT_NO_DIFFERENT_CHARS = 0.2  # Reduced to maintain balance
MIN_CHARS = 2  # Minimum chord length
CORPUS_FILE_NAME = "MonkeyType_english_10k.json"


@dataclass
class OptimizationMetrics:
    """Stores metrics about the optimization process"""

    total_cost: float
    approximation_ratio: float
    character_similarity: float
    first_last_usage: float
    unassigned_count: int
    average_chord_length: float


def calculate_harmonic_number(n: int) -> float:
    """Calculate the nth harmonic number"""
    return sum(1 / k for k in range(1, n + 1))


def get_word_weight(rank: int, total_words: int) -> float:
    """Calculate Zipf's law weight for a word using proper harmonic normalization"""
    h_n = calculate_harmonic_number(total_words)
    return 1.0 / ((rank + 1) * h_n)


def has_first_last_bonus(word: str, chord: str) -> float:
    """Calculate bonus for including first/last characters of word in chord"""
    if chord == "EMPTY":
        return 0.0
    bonus = 0.0
    if word[0] in chord:
        bonus += 0.1
    if word[-1] in chord:
        bonus += 0.1
    return bonus


def different_chars_penalty(word: str, chord: str) -> float:
    """Calculate penalty for characters in chord not in word"""
    if chord == "EMPTY":
        return 0.0
    return len([c for c in chord if c not in word])


def get_valid_combinations(word: str) -> List[str]:
    """Generate all valid combinations of letters from a word."""
    letters = sorted(set(word.lower()))
    valid_combos = []
    for r in range(2, min(len(letters) + 1, MAX_CHARS + 1)):
        combos = combinations(letters, r)
        valid_combos.extend("".join(combo) for combo in combos)

    return sorted(valid_combos, key=len)  # Sort by length for optimization


def get_weighted_combinations(
    word: str, rank: int, total_words: int
) -> List[Tuple[str, float]]:
    """Generate combinations with their weighted costs"""
    combos = get_valid_combinations(word)
    weight = get_word_weight(rank, total_words)
    weighted_combos = []

    for combo in combos:
        # Base cost is always positive and scales with chord length
        base_cost = len(combo) * weight
        # Bonuses and penalties are scaled relative to base cost
        first_last_bonus = (
            WEIGHT_FIRST_LAST_CHAR * has_first_last_bonus(word, combo) * base_cost
        )
        diff_chars_penalty = (
            WEIGHT_NO_DIFFERENT_CHARS * different_chars_penalty(word, combo) * base_cost
        )

        cost = base_cost - first_last_bonus + diff_chars_penalty
        weighted_combos.append((combo, cost))

    return weighted_combos


def check_assignment_against_rules(word: str, chord: str) -> bool:
    """Check chord assignment against optimization rules"""
    if chord == "EMPTY":
        return True
    if not 2 <= len(chord) <= MAX_CHARS:
        return False
    return all(character in word for character in chord)


def calculate_total_cost(assignments: Dict[str, str], words: List[str]) -> float:
    """Calculate total weighted cost of assignments"""
    total_cost = 0.0
    total_words = len(words)

    for rank, word in enumerate(words):
        chord = assignments[word]
        if chord != "EMPTY":
            weight = get_word_weight(rank, total_words)
            # Base cost is always positive and scales with chord length
            base_cost = len(chord) * weight
            # Bonuses and penalties are scaled relative to base cost
            first_last_bonus = (
                WEIGHT_FIRST_LAST_CHAR * has_first_last_bonus(word, chord) * base_cost
            )
            diff_chars_penalty = (
                WEIGHT_NO_DIFFERENT_CHARS
                * different_chars_penalty(word, chord)
                * base_cost
            )

            cost = base_cost - first_last_bonus + diff_chars_penalty
            total_cost += cost

    return total_cost


def get_theoretical_lower_bound(words: List[str]) -> float:
    """Calculate theoretical lower bound assuming optimal conditions"""
    total_words = len(words)
    return sum(2 * get_word_weight(rank, total_words) for rank in range(total_words))


def calculate_character_similarity(assignments: Dict[str, str]) -> float:
    """Calculate average character similarity between words and their chords"""
    similarities = []
    for word, chord in assignments.items():
        if chord != "EMPTY":
            word_chars = set(word)
            chord_chars = set(chord)
            similarity = len(chord_chars.intersection(word_chars)) / len(chord_chars)
            similarities.append(similarity)
    return sum(similarities) / len(similarities) if similarities else 0.0


def calculate_first_last_usage(assignments: Dict[str, str]) -> float:
    """Calculate proportion of assignments using first/last characters"""
    usage_count = 0
    total = 0
    for word, chord in assignments.items():
        if chord != "EMPTY":
            total += 1
            if word[0] in chord or word[-1] in chord:
                usage_count += 1
    return usage_count / total if total > 0 else 0.0


def assign_chords(words: List[str]) -> Tuple[Dict[str, str], OptimizationMetrics]:
    """Assign chords using weighted optimization approach"""
    used_chords: Set[str] = set()
    assignments: Dict[str, str] = {}
    unassigned_count: int = 0
    total_length: int = 0
    assigned_count: int = 0
    total_words = len(words)

    for rank, word in enumerate(words):
        weighted_combos = get_weighted_combinations(word, rank, total_words)
        chord_assigned = False

        # Try to assign the combination with lowest weighted cost
        for combo, _ in sorted(weighted_combos, key=lambda x: x[1]):
            if combo not in used_chords and check_assignment_against_rules(word, combo):
                assignments[word] = combo
                used_chords.add(combo)
                chord_assigned = True
                total_length += len(combo)
                assigned_count += 1
                break

        if not chord_assigned:
            assignments[word] = "EMPTY"
            unassigned_count += 1

    # Calculate optimization metrics
    total_cost = calculate_total_cost(assignments, words)
    lower_bound = get_theoretical_lower_bound(words)
    approximation_ratio = total_cost / lower_bound if lower_bound > 0 else float("inf")
    character_similarity = calculate_character_similarity(assignments)
    first_last_usage = calculate_first_last_usage(assignments)
    average_chord_length = total_length / assigned_count if assigned_count > 0 else 0

    metrics = OptimizationMetrics(
        total_cost=total_cost,
        approximation_ratio=approximation_ratio,
        character_similarity=character_similarity,
        first_last_usage=first_last_usage,
        unassigned_count=unassigned_count,
        average_chord_length=average_chord_length,
    )

    return assignments, metrics


def process_corpus_json(input_file_name: str):
    """Process a corpus JSON file and create optimized chord assignments."""
    with open(input_file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data["words"]
    assignments, metrics = assign_chords(words)

    # Print optimization metrics
    print("\nOptimization Metrics:")
    print(f"Total Weighted Cost: {metrics.total_cost:.4f}")
    print(f"Approximation Ratio: {metrics.approximation_ratio:.4f}")
    print(f"Character Similarity: {metrics.character_similarity:.4f}")
    print(f"First/Last Usage: {metrics.first_last_usage:.4f}")
    print(f"Unassigned Words: {metrics.unassigned_count}")
    print(f"Average Chord Length: {metrics.average_chord_length:.2f}")

    output_data = {
        "name": "optimized_chords_for_" + data["name"],
        "orderedByFrequency": data["orderedByFrequency"],
        "optimizationMetrics": {
            "totalCost": metrics.total_cost,
            "approximationRatio": metrics.approximation_ratio,
            "characterSimilarity": metrics.character_similarity,
            "firstLastUsage": metrics.first_last_usage,
            "unassignedCount": metrics.unassigned_count,
            "averageChordLength": metrics.average_chord_length,
        },
        "chords": [assignments[word] + " -> " + word for word in words],
    }

    with open("OptimizedChordsFor_" + input_file_name, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    process_corpus_json(CORPUS_FILE_NAME)
