"""
Chord generator using Zipf's law with customizable weights
"""

import argparse
import functools
import json
import string
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

from config import GeneratorConfig


@dataclass
class OptimizationMetrics:
    """Stores metrics about the optimization process"""

    total_cost: float
    approximation_ratio: float
    first_last_usage: float
    fallback_assignments: int
    average_chord_length: float
    single_letter_words: int


def create_keyboard_mapping(csv_path: Path) -> Dict[str, Tuple[int, int]]:
    """Creates a mapping of letters to (finger, row) tuples from a CSV keyboard layout"""
    finger_map = [0, 1, 2, 3, 3, 4, 4, 5, 6, 7]  # Maps column index to finger
    mapping = {}

    with open(csv_path, encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f]
        for row_idx, row in enumerate(reversed(rows)):  # Process bottom-up
            for col_idx, letter in enumerate(row):
                if letter != "-":
                    mapping[letter] = (finger_map[col_idx], row_idx)

    return mapping


# Cache harmonic numbers for performance
@functools.lru_cache(maxsize=1000)
def calculate_harmonic_number(n: int) -> float:
    """Calculate the nth harmonic number"""
    return sum(1 / k for k in range(1, n + 1))


def get_zipf_word_weight(rank: int, total_words: int) -> float:
    """Calculate Zipf's law weight using proper harmonic normalization"""
    if total_words <= 0:
        raise ValueError("Total words must be positive")
    h_n = calculate_harmonic_number(total_words)
    return 1.0 / ((rank + 1) * h_n)


def calculate_assignment_cost(
    word: str,
    word_rank: int,
    total_word_count: int,
    suggested_chord: str,
    config: GeneratorConfig,
    fallback_iteration: int = 0,
) -> float:
    """Calculate weighted cost of a chord assignment based on config preferences"""
    if len(word) == 1:
        return 0.0
    if not suggested_chord:
        raise ValueError("Empty chord for multi-letter word")

    zipf_word_weight = get_zipf_word_weight(word_rank, total_word_count)
    cost_factor = (
        len(suggested_chord)
        * config.chord_length_weight
        * (config.first_letter_weight if word[0].lower() not in suggested_chord else 1)
        * (
            config.second_letter_weight
            if len(word) > 1 and word[1].lower() not in suggested_chord
            else 1
        )
        * (config.last_letter_weight if word[-1].lower() not in suggested_chord else 1)
        * (
            config.fallback_penalty * fallback_iteration
            if fallback_iteration > 0
            else 1
        )
    )
    return zipf_word_weight * cost_factor


def generate_standard_chords(
    word: str, unavailable_chords: Set[str], config: GeneratorConfig
) -> List[str]:
    """Generate all possible chords with letters from the word that are not yet in use"""
    letters = sorted(set(word.lower()))
    valid_combos: List[str] = []

    # Generate all possible chord combinations from MIN_LETTERS to MAX_LETTERS
    max_length = min(len(letters) + 1, config.max_letters + 1)
    for length in range(config.min_letters, max_length):
        possible_combinations = combinations(letters, length)
        for combo in possible_combinations:
            chord = "".join(combo)
            if chord not in unavailable_chords:
                valid_combos.append(chord)

    return sorted(valid_combos, key=len)  # sorted small to large


def get_letter_word_compatibility(
    letter: str, word: str, layout: Dict[str, Tuple[int, int]]
) -> float:
    """
    Calculate compatibility score of adding a letter to a word based on keyboard layout.
    Returns a score between 0.0 (worst) and 1.0 (best).

    Args:
        letter: The letter to evaluate
        word: The current word
        layout: Dictionary mapping letters to (finger, row) tuples
    """
    if not word or letter not in layout:
        return 1.0

    letter_finger, letter_row = layout[letter]
    score = 1.0

    # Get fingers used in word
    used_fingers = {layout[c][0] for c in word if c in layout}

    # Immediate disqualification for same-finger usage
    if letter_finger in used_fingers:
        return 0.0

    # Check lateral adjacency with last letter
    if word[-1] in layout:
        last_finger, last_row = layout[word[-1]]
        finger_distance = abs(letter_finger - last_finger)

        # Penalize horizontally adjacent keys (unless in different rows)
        if finger_distance == 1 and letter_row == last_row:
            score -= 0.3

        # Reward alternating hands (fingers 0-3 left hand, 4-7 right hand)
        if (letter_finger < 4) != (last_finger < 4):
            score += 0.2

        # Small penalty for large vertical jumps
        row_difference = abs(letter_row - last_row)
        if row_difference > 1:
            score -= 0.1 * row_difference

    return max(0.0, min(1.0, score))


# ,Use set operations for better performance
def get_most_compatible_letter(word: str, layout: Dict[str, Tuple[int, int]]) -> str:
    """Find the most compatible letter to add to a word based on keyboard layout"""
    unused_letters = set(string.ascii_lowercase) - set(word.lower())
    return max(
        unused_letters,
        key=lambda letter: get_letter_word_compatibility(letter, word, layout),
    )


# TODO: check lower() usage
def generate_fallback_chords(
    word: str,
    layout: dict[str, tuple[int, int]],
    unavailable_chords: Set[str],
    config: GeneratorConfig,
    fallback_iteration: int,
) -> Tuple[List[str], int]:
    """Generate a fallback chord that also contains letters not in the word"""
    fallback_iteration += 1
    new_word = word + get_most_compatible_letter(word, layout)
    new_chords = generate_standard_chords(new_word, unavailable_chords, config)
    if not new_chords:
        return generate_fallback_chords(
            new_word, layout, unavailable_chords, config, fallback_iteration
        )
    return new_chords, fallback_iteration


def assign_chords(
    words: List[str], layout: Dict[str, Tuple[int, int]], config: GeneratorConfig
) -> Tuple[Dict[str, str], OptimizationMetrics]:
    """Assign chords using weighted optimization with guaranteed assignments"""
    if not words:
        raise ValueError("Empty word list")

    total_words = len(words)
    used_chords: Set[str] = set()
    assignments: Dict[str, str] = {}
    metrics = {
        "total_length": 0,
        "single_letter_count": 0,
        "fallback_count": 0,
        "words_with_chords": 0,
        "total_cost": 0.0,
    }

    for rank, word in enumerate(words):
        if len(word) == 1:
            assignments[word] = ""
            metrics["single_letter_count"] += 1
            continue

        valid_combos = generate_standard_chords(word, used_chords, config)
        fallback_count = 0
        if not valid_combos:
            valid_combos, fallback_count = generate_fallback_chords(
                word, layout, used_chords, config, fallback_iteration=0
            )

        combo_costs = [
            (
                calculate_assignment_cost(
                    word, rank, total_words, combo, config, fallback_count
                ),
                combo,
            )
            for combo in valid_combos
        ]
        cost, best_combo = min(combo_costs)

        assignments[word] = best_combo
        used_chords.add(best_combo)
        metrics["total_length"] += len(best_combo)
        metrics["words_with_chords"] += 1
        metrics["total_cost"] += cost

    # Calculate metrics
    words_with_actual_chords = [(w, c) for w, c in assignments.items() if c]

    lower_bound = sum(
        config.min_letters * get_zipf_word_weight(rank, total_words)
        for rank, word in enumerate(words)
        if len(word) > 1
    )

    return assignments, OptimizationMetrics(
        total_cost=metrics["total_cost"],
        approximation_ratio=(
            metrics["total_cost"] / lower_bound if lower_bound > 0 else 1.0
        ),
        first_last_usage=(
            sum(1 for w, c in words_with_actual_chords if w[0] in c or w[-1] in c)
            / len(words_with_actual_chords)
            if words_with_actual_chords
            else 1.0
        ),
        fallback_assignments=metrics["fallback_count"],
        average_chord_length=(
            metrics["total_length"] / metrics["words_with_chords"]
            if metrics["words_with_chords"] > 0
            else 0
        ),
        single_letter_words=metrics["single_letter_count"],
    )


def process_corpus_json(config_path: Path) -> None:
    """Process corpus and generate optimized chord assignments"""
    # Load configuration
    config = GeneratorConfig.load_config(config_path)
    config.validate()

    # Load corpus
    try:
        with open(config.corpus_json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        sys.exit(f"Error reading input file: {e}")

    if data.get("orderedByFrequency") is False:
        sys.exit("List not ordered by frequency")

    print("Processing word list...")
    words = data.get("words", [])
    if not words:
        sys.exit("Empty word list in input file")

    layout = create_keyboard_mapping(Path(config.keylayout_csv_file))

    print("Assigning chords...")
    try:
        assignments, metrics = assign_chords(words, layout, config)
    except RuntimeError as e:
        sys.exit(f"Failed to generate chords: {str(e)}")

    output_data = {
        "name": "optimized_chords_for_" + data.get("name", "unknown"),
        "orderedByFrequency": data.get("orderedByFrequency", True),
        "optimizationMetrics": {
            "totalCost": metrics.total_cost,
            "approximationRatio": metrics.approximation_ratio,
            "firstLastUsage": metrics.first_last_usage,
            "fallbackAssignments": metrics.fallback_assignments,
            "averageChordLength": metrics.average_chord_length,
            "singleLetterWords": metrics.single_letter_words,
        },
        "chords": [
            (
                (word + " -> " + assignments[word])
                if word in assignments and assignments[word]
                else word
            )
            for word in words
        ],
    }

    output_filename = f"ChordsFor_{config.corpus_json_file}"
    try:
        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=2)
        print(f"\nOutput written to {output_filename}")

        print("\nOptimization Metrics:")
        print(f"Total Weighted Cost: {metrics.total_cost:.4f}")
        print(f"Approximation Ratio: {metrics.approximation_ratio:.4f}")
        print(f"First/Last Usage: {metrics.first_last_usage:.4f}")
        print(f"Fallback Assignments: {metrics.fallback_assignments}")
        print(f"Average Chord Length: {metrics.average_chord_length:.2f}")
        print(f"Single letter Words: {metrics.single_letter_words}")
    except IOError as e:
        sys.exit(f"Error writing output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a corpus JSON file.")
    parser.add_argument(
        "corpus_file",
        nargs="?",
        type=str,
        help="Path to the corpus JSON file (overrides config file setting)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("generator.config"),
        help="Path to configuration file (default: generator.config)",
    )

    args = parser.parse_args()

    # If corpus file is provided, modify the config to use it
    config = GeneratorConfig.load_config(args.config)
    if args.corpus_file:
        config.corpus_json_file = args.corpus_file

    process_corpus_json(args.config)
