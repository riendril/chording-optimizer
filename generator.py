"""
This module generates an optimized set of chords for a set of words following Zipf's law
and a set of rules to minimize average typing effort.
"""

import json
from dataclasses import dataclass
from itertools import combinations
from math import log
from typing import Dict, List, Set, Tuple

CORPUS_FILE_NAME = "MonkeyType_english_10k.json"


@dataclass
class OptimizationMetrics:
    """Stores metrics about the optimization process"""

    total_cost: float
    approximation_ratio: float
    unassigned_count: int
    average_chord_length: float


def get_word_weight(rank: int) -> float:
    """Calculate Zipf's law weight for a word.
    f(k) = 1/(k * ln(1.78 + n))
    where k is rank and n is vocabulary size"""
    return 1.0 / ((rank + 1) * log(1.78 + rank))


def get_valid_combinations(word: str) -> List[str]:
    """Generate all valid combinations of letters from a word."""
    letters = sorted(set(word.lower()))
    valid_combos = []
    for r in range(2, len(letters) + 1):
        combos = combinations(letters, r)
        valid_combos.extend("".join(combo) for combo in combos)

    return sorted(valid_combos, key=len)  # Sort by length for optimization


def get_weighted_combinations(word: str, rank: int) -> List[Tuple[str, float]]:
    """Generate combinations with their costs weighted by frequency"""
    combos = get_valid_combinations(word)
    weight = get_word_weight(rank)
    return [(combo, len(combo) * weight) for combo in combos]


def check_assignment_against_rules(word: str, chord: str) -> bool:
    """Check chord assignment against optimization rules"""
    if len(chord) >= 2 and len(chord) < 6:
        return all(character in word for character in chord)
    return False


def calculate_total_cost(assignments: Dict[str, str], words: List[str]) -> float:
    """Calculate total weighted cost of assignments"""
    total_cost = 0.0
    for rank, word in enumerate(words):
        chord = assignments[word]
        if chord != "EMPTY":
            weight = get_word_weight(rank)
            total_cost += weight * len(chord)
    return total_cost


def get_theoretical_lower_bound(words: List[str]) -> float:
    """Calculate theoretical lower bound assuming optimal conditions"""
    lower_bound = 0.0
    for rank, _ in enumerate(words):
        weight = get_word_weight(rank)
        lower_bound += weight * 2  # minimum chord length is 2
    return lower_bound


def assign_chords(words: List[str]) -> Tuple[Dict[str, str], OptimizationMetrics]:
    """Assign chords using an optimized approach considering Zipf's law weights"""
    used_chords: Set[str] = set()
    assignments: Dict[str, str] = {}
    unassigned_count: int = 0
    total_length: int = 0
    assigned_count: int = 0

    for rank, word in enumerate(words):
        weighted_combos = get_weighted_combinations(word, rank)
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
    average_chord_length = total_length / assigned_count if assigned_count > 0 else 0

    metrics = OptimizationMetrics(
        total_cost=total_cost,
        approximation_ratio=approximation_ratio,
        unassigned_count=unassigned_count,
        average_chord_length=average_chord_length,
    )

    return assignments, metrics


def process_corpus_json(input_file_name: str):
    """Process a corpus JSON file ALREADY SORTED BY FREQUENCY and create
    a new one with optimized chord assignments."""
    with open(input_file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data["words"]
    assignments, metrics = assign_chords(words)

    # Print optimization metrics
    print(f"\nOptimization Metrics:")
    print(f"Total Weighted Cost: {metrics.total_cost:.4f}")
    print(f"Approximation Ratio: {metrics.approximation_ratio:.4f}")
    print(f"Unassigned Words: {metrics.unassigned_count}")
    print(f"Average Chord Length: {metrics.average_chord_length:.2f}")

    output_data = {
        "name": "optimized_chords_for_" + data["name"],
        "orderedByFrequency": data["orderedByFrequency"],
        "optimizationMetrics": {
            "totalCost": metrics.total_cost,
            "approximationRatio": metrics.approximation_ratio,
            "unassignedCount": metrics.unassigned_count,
            "averageChordLength": metrics.average_chord_length,
        },
        "chords": [assignments[word] + " -> " + word for word in words],
    }

    with open("OptimizedChordsFor_" + input_file_name, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    process_corpus_json(CORPUS_FILE_NAME)
