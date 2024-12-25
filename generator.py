"""
Improved chord generator with guaranteed assignments and normalized cost function.
"""

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, Tuple

# Configuration constants
MAX_CHARS = 5
WEIGHT_FIRST_LAST_CHAR = 0.3
WEIGHT_NO_DIFFERENT_CHARS = 0.4
MIN_CHARS = 2
FALLBACK_PENALTY = 1.5  # Penalty multiplier for fallback assignments


@dataclass
class OptimizationMetrics:
    """Stores metrics about the optimization process"""

    total_cost: float
    approximation_ratio: float
    character_similarity: float
    first_last_usage: float
    fallback_assignments: int
    average_chord_length: float


def calculate_harmonic_number(n: int) -> float:
    """Calculate the nth harmonic number"""
    return sum(1 / k for k in range(1, n + 1))


def get_word_weight(rank: int, total_words: int) -> float:
    """Calculate Zipf's law weight using proper harmonic normalization"""
    h_n = calculate_harmonic_number(total_words)
    return 1.0 / ((rank + 1) * h_n)


def similarity_score(word: str, chord: str) -> float:
    """Calculate normalized similarity score between word and chord"""
    word_chars = set(word.lower())
    chord_chars = set(chord.lower())
    intersection = len(chord_chars.intersection(word_chars))
    return intersection / len(chord)


def get_valid_combinations(word: str, used_chords: Set[str]) -> List[str]:
    """Generate all valid combinations of letters from a word."""
    letters = sorted(set(word.lower()))
    valid_combos = []

    # First try combinations using word letters
    for r in range(MIN_CHARS, min(len(letters) + 1, MAX_CHARS + 1)):
        combos = combinations(letters, r)
        valid_combos.extend(
            "".join(combo) for combo in combos if "".join(combo) not in used_chords
        )

    return sorted(valid_combos, key=len)


def get_fallback_chord(word: str, used_chords: Set[str]) -> str:
    """Generate a fallback chord when no optimal combination is available"""
    word = word.lower()
    # Try to use first/last characters first
    base_chars = [word[0]]
    if len(word) > 1:
        base_chars.append(word[-1])

    # Add unique middle characters until we have enough
    middle_chars = sorted(set(word[1:-1])) if len(word) > 2 else []

    for length in range(MIN_CHARS, MAX_CHARS + 1):
        for middle_combo in combinations(
            middle_chars, max(0, length - len(base_chars))
        ):
            chord = "".join(base_chars + list(middle_combo))
            if chord not in used_chords:
                return chord

    # If still no chord found, create one using character substitution
    for c in "abcdefghijklmnopqrstuvwxyz":
        for i in range(len(word) - 1):
            chord = word[i] + c
            if chord not in used_chords:
                return chord

    return ""  # Should never reach here if MAX_CHARS > 1


def calculate_assignment_cost(
    word: str, chord: str, rank: int, total_words: int, is_fallback: bool = False
) -> float:
    """Calculate normalized cost for a chord assignment"""
    weight = get_word_weight(rank, total_words)
    base_cost = len(chord) * weight

    # Normalize similarity to [0, 1]
    sim_score = similarity_score(word, chord)

    # Apply fallback penalty if this is a fallback assignment
    if is_fallback:
        base_cost *= FALLBACK_PENALTY

    return base_cost * (2 - sim_score)  # Higher similarity reduces cost


def assign_chords(words: List[str]) -> Tuple[Dict[str, str], OptimizationMetrics]:
    """Assign chords using weighted optimization with guaranteed assignments"""
    used_chords: Set[str] = set()
    assignments: Dict[str, str] = {}
    fallback_count: int = 0
    total_length: int = 0
    total_words = len(words)

    # First pass: optimal assignments
    for rank, word in enumerate(words):
        valid_combos = get_valid_combinations(word, used_chords)
        assigned = False

        if valid_combos:
            # Find combo with minimum cost
            costs = [
                (calculate_assignment_cost(word, combo, rank, total_words), combo)
                for combo in valid_combos
            ]
            cost, best_combo = min(costs)

            assignments[word] = best_combo
            used_chords.add(best_combo)
            total_length += len(best_combo)
            assigned = True

        if not assigned:
            # Use fallback assignment
            fallback = get_fallback_chord(word, used_chords)
            if fallback:
                assignments[word] = fallback
                used_chords.add(fallback)
                total_length += len(fallback)
                fallback_count += 1

    # Calculate metrics
    total_cost = sum(
        calculate_assignment_cost(
            word, chord, rank, total_words, chord == assignments[word]
        )
        for rank, (word, chord) in enumerate(assignments.items())
    )

    # Calculate theoretical lower bound (all words with minimum length chords)
    lower_bound = sum(
        MIN_CHARS * get_word_weight(rank, total_words) for rank in range(total_words)
    )

    metrics = OptimizationMetrics(
        total_cost=total_cost,
        approximation_ratio=total_cost / lower_bound,
        character_similarity=sum(similarity_score(w, c) for w, c in assignments.items())
        / len(words),
        first_last_usage=sum(
            1 for w, c in assignments.items() if w[0] in c or w[-1] in c
        )
        / len(words),
        fallback_assignments=fallback_count,
        average_chord_length=total_length / len(words),
    )

    return assignments, metrics


def process_corpus_json(input_file_name: str):
    """Process corpus and generate optimized chord assignments"""
    with open(input_file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data["words"]
    assignments, metrics = assign_chords(words)

    print("\nOptimization Metrics:")
    print(f"Total Weighted Cost: {metrics.total_cost:.4f}")
    print(f"Approximation Ratio: {metrics.approximation_ratio:.4f}")
    print(f"Character Similarity: {metrics.character_similarity:.4f}")
    print(f"First/Last Usage: {metrics.first_last_usage:.4f}")
    print(f"Fallback Assignments: {metrics.fallback_assignments}")
    print(f"Average Chord Length: {metrics.average_chord_length:.2f}")

    output_data = {
        "name": "optimized_chords_for_" + data["name"],
        "orderedByFrequency": data["orderedByFrequency"],
        "optimizationMetrics": {
            "totalCost": metrics.total_cost,
            "approximationRatio": metrics.approximation_ratio,
            "characterSimilarity": metrics.character_similarity,
            "firstLastUsage": metrics.first_last_usage,
            "fallbackAssignments": metrics.fallback_assignments,
            "averageChordLength": metrics.average_chord_length,
        },
        "chords": [assignments[word] + " -> " + word for word in words],
    }

    with open("OptimizedChordsFor_" + input_file_name, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    process_corpus_json("MonkeyType_english_10k.json")
