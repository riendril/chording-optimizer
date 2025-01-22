"""
Chord generator using Zipf's law with customizable weights
"""

import argparse
import cProfile
import functools
import io
import json
import pstats
import string
import sys
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.get_parameters import GeneratorConfig


class FingerIndex(Enum):
    LPINKY = 2
    LRING = 3
    LMIDDLE = 4
    LINDEX = 5
    LTHUMB = 6
    RTHUMB = 7
    RINDEX = 8
    RMIDDLE = 9
    RRING = 10
    RPINKY = 11


@dataclass
class WordData:
    """Store preprocessed word data to avoid repeated operations"""

    original: str
    lower: str
    length: int

    @classmethod
    def from_word(cls, word: str) -> "WordData":
        return cls(original=word, lower=word.lower(), length=len(word))


@dataclass
class LetterData:
    """Store relevant information about a letter"""

    finger: FingerIndex
    vertical_distance_to_home_row: int
    horizontal_distance_to_home_row: int
    finger_to_left: Optional[FingerIndex]
    finger_to_right: Optional[FingerIndex]


@dataclass
class OptimizationMetrics:
    """Store metrics about the optimization process"""

    total_cost: float
    first_last_usage: float
    fallback_assignments: int
    average_chord_length: float
    single_letter_words: int


def profile_function(func):
    """Decorator to profile a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Print top 20 time-consuming operations
        print(f"\nProfile for {func.__name__}:")
        print(s.getvalue())
        return result

    return wrapper


def create_keyboard_mapping(config: GeneratorConfig) -> Dict[str, LetterData]:
    """Creates a mapping of letters to (finger, row) tuples from a CSV keyboard layout"""

    mapping = {}

    # Create finger mapping dictionary
    finger_mapping = {
        "lp": FingerIndex.LPINKY,
        "lr": FingerIndex.LRING,
        "lm": FingerIndex.LMIDDLE,
        "li": FingerIndex.LINDEX,
        "ri": FingerIndex.RINDEX,
        "rm": FingerIndex.RMIDDLE,
        "rr": FingerIndex.RRING,
        "rp": FingerIndex.RPINKY,
    }
    with open(config.keylayout_csv_file, encoding="utf-8") as file:
        rows = [line.strip().split(",") for line in file]
        # First 3 rows are the keyboard layout
        layout_rows = rows[:3]
        # Next 3 rows are finger mappings
        finger_map_rows = rows[3:6]
        # Next 3 rows are vertical mappings
        vertical_map_rows = rows[6:9]
        # Last 3 rows are horizontal mappings
        horizontal_map_rows = rows[9:12]
        if config.keylayout_type == "matrix":
            # Pre-compute sets for boundary checks
            no_left_fingers = {FingerIndex.LPINKY, FingerIndex.RINDEX}
            no_right_fingers = {FingerIndex.RPINKY, FingerIndex.LINDEX}
        elif config.keylayout_type == "M4G":
            # Pre-compute sets for boundary checks
            no_left_fingers = {FingerIndex.LPINKY, FingerIndex.RTHUMB}
            no_right_fingers = {FingerIndex.RPINKY, FingerIndex.LTHUMB}
        else:
            sys.exit("Invalid LayoutType")
        for row_idx, row in enumerate(layout_rows):
            for col_idx, letter in enumerate(row):
                if letter == "-":
                    continue
                current_finger = finger_mapping[finger_map_rows[row_idx][col_idx]]
                mapping[letter] = LetterData(
                    current_finger,
                    int(vertical_map_rows[row_idx][col_idx]),
                    int(horizontal_map_rows[row_idx][col_idx]),
                    (
                        None
                        if current_finger in no_left_fingers
                        else finger_mapping[finger_map_rows[row_idx][col_idx - 1]]
                    ),
                    (
                        None
                        if current_finger in no_right_fingers
                        else finger_mapping[finger_map_rows[row_idx][col_idx + 1]]
                    ),
                )
    return mapping


# Cache harmonic number for performance
@functools.lru_cache(maxsize=1)
def calculate_harmonic_number(n: int) -> float:
    """Calculate the nth harmonic number"""
    return sum(1 / k for k in range(1, n + 1))


# No cache needed since each rank is only used once
def get_zipf_word_weight(rank: int, total_words: int) -> float:
    """Calculate Zipf's law weight using proper harmonic normalization"""
    if total_words <= 0:
        raise ValueError("Total words must be positive")
    h_n = calculate_harmonic_number(total_words)
    return 1.0 / ((rank + 1) * h_n)


def calculate_assignment_cost(
    word: WordData,
    word_rank: int,
    total_word_count: int,
    suggested_chord: str,
    config: GeneratorConfig,
    fallback_iteration: int = 0,
) -> float:
    """Calculate weighted cost of a chord assignment based on config preferences"""
    if word.length == 1:
        return 0.0

    zipf_word_weight = get_zipf_word_weight(word_rank, total_word_count)

    cost_factor = (
        config.additional_letter_weight ** len(suggested_chord)
        * (
            config.first_letter_unmatched_weight
            if word.lower[0] not in suggested_chord
            else 1
        )
        * (
            config.second_letter_unmatched_weight
            if word.length > 1 and word.lower[1] not in suggested_chord
            else 1
        )
        * (
            config.last_letter_unmatched_weight
            if word.lower[-1] not in suggested_chord
            else 1
        )
        * (
            config.fallback_letter_weight * fallback_iteration
            if fallback_iteration > 0
            else 1
        )
    )
    return zipf_word_weight * cost_factor


def generate_standard_chords(
    word_data: WordData, unavailable_chords: Set[str], config: GeneratorConfig
) -> List[str]:
    """Generate all possible chords with letters from the word that are not yet in use"""
    letters = set(word_data.lower)
    valid_combos: List[str] = []

    max_length = min(len(letters) + 1, config.max_letters + 1)
    for length in range(config.min_letters, max_length):
        possible_combinations = combinations(letters, length)
        for combo in possible_combinations:
            chord = "".join(combo)
            if chord not in unavailable_chords:
                valid_combos.append(chord)

    return valid_combos


def get_letter_word_compatibility_cost(
    letter: str,
    word_data: WordData,
    layout: Dict[str, LetterData],
    config: GeneratorConfig,
) -> float:
    """Calculate compatibility cost of adding a letter to a word based on keyboard layout"""

    letter_data = layout[letter]
    cost = 1.0

    # Get fingers used in word and their positions
    word_fingers = []
    word_positions = []
    for c in word_data.lower:
        word_fingers.append(layout[c].finger)
        word_positions.append(
            (
                layout[c].vertical_distance_to_home_row,
                layout[c].horizontal_distance_to_home_row,
            )
        )

    # Account for vertical movements
    if letter_data.vertical_distance_to_home_row != 0:
        # Check for vertical stretches (non-adjacent rows)
        if abs(letter_data.vertical_distance_to_home_row) > 1:
            cost *= config.vertical_stretch_weight
        # Check for vertical pinches with previous letters
        for pos in word_positions:
            if pos[0] != 0 and (pos[0] * letter_data.vertical_distance_to_home_row) < 0:
                cost *= config.vertical_pinch_weight

    # Account for horizontal movements
    if letter_data.horizontal_distance_to_home_row != 0:
        # Check for horizontal stretches
        if letter_data.finger_to_left in word_fingers:
            cost *= config.horizontal_stretch_weight
        if letter_data.finger_to_right in word_fingers:
            cost *= config.horizontal_stretch_weight
        # Check for horizontal pinches
        for pos in word_positions:
            if (
                pos[1] != 0
                and (pos[1] * letter_data.horizontal_distance_to_home_row) < 0
            ):
                cost *= config.horizontal_pinch_weight

    # Account for diagonal movements
    if (
        letter_data.vertical_distance_to_home_row != 0
        and letter_data.horizontal_distance_to_home_row != 0
    ):
        # Check for diagonal stretches
        for pos in word_positions:
            if pos[0] != 0 and pos[1] != 0:
                cost *= config.diagonal_stretch_weight
                # Check for diagonal pinches
                if (
                    pos[0] * letter_data.vertical_distance_to_home_row < 0
                    or pos[1] * letter_data.horizontal_distance_to_home_row < 0
                ):
                    cost *= config.diagonal_pinch_weight

    # Account for same finger usage
    if word_fingers:
        last_finger = word_fingers[-1]
        if letter_data.finger == last_finger:
            # Check for doubles
            cost *= config.same_finger_double_weight
            # Check for triples
            if len(word_fingers) >= 2 and word_fingers[-2] == last_finger:
                cost *= config.same_finger_triple_weight

    # Account for specific awkward combinations
    if word_fingers:
        last_finger = word_fingers[-1]
        curr_finger = letter_data.finger

        # Pinky-ring stretches
        if (
            last_finger in (FingerIndex.LPINKY, FingerIndex.RPINKY)
            and curr_finger in (FingerIndex.LRING, FingerIndex.RRING)
        ) or (
            last_finger in (FingerIndex.LRING, FingerIndex.RRING)
            and curr_finger in (FingerIndex.LPINKY, FingerIndex.RPINKY)
        ):
            cost *= config.pinky_ring_stretch_weight

        # Ring-middle scissors
        if (
            last_finger in (FingerIndex.LRING, FingerIndex.RRING)
            and curr_finger in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
        ) or (
            last_finger in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
            and curr_finger in (FingerIndex.LRING, FingerIndex.RRING)
        ):
            cost *= config.ring_middle_scissor_weight

        # Middle-index stretches
        if (
            last_finger in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
            and curr_finger in (FingerIndex.LINDEX, FingerIndex.RINDEX)
        ) or (
            last_finger in (FingerIndex.LINDEX, FingerIndex.RINDEX)
            and curr_finger in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
        ):
            cost *= config.middle_index_stretch_weight

    return cost


# ,Use set operations for better performance
def get_most_compatible_letter(
    word: WordData, layout: Dict[str, LetterData], config: GeneratorConfig
) -> str:
    """Find the most compatible letter to add to a word based on keyboard layout"""
    unused_letters = set(string.ascii_lowercase) - set(word.lower)
    return min(
        unused_letters,
        key=lambda letter: get_letter_word_compatibility_cost(
            letter, word, layout, config
        ),
    )


def generate_fallback_chords(
    word: WordData,
    layout: dict[str, LetterData],
    unavailable_chords: Set[str],
    config: GeneratorConfig,
    fallback_iteration: int,
) -> Tuple[List[str], int]:
    """Generate a fallback chord that also contains letters not in the word"""
    fallback_iteration += 1
    new_letter = get_most_compatible_letter(word, layout, config)
    new_word = word
    new_word.length += 1
    new_word.lower += new_letter
    new_word.original += new_letter
    new_chords = generate_standard_chords(new_word, unavailable_chords, config)
    if not new_chords:
        return generate_fallback_chords(
            new_word, layout, unavailable_chords, config, fallback_iteration
        )
    return new_chords, fallback_iteration


def assign_chords(
    words: List[str], layout: Dict[str, LetterData], config: GeneratorConfig
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
        word_data = WordData.from_word(word)
        if word_data.length == 1:
            assignments[word] = ""
            metrics["single_letter_count"] += 1
            continue

        valid_combos = generate_standard_chords(word_data, used_chords, config)
        fallback_count = 0
        if not valid_combos:
            valid_combos, fallback_count = generate_fallback_chords(
                word_data, layout, used_chords, config, fallback_iteration=0
            )

        combo_costs = [
            (
                calculate_assignment_cost(
                    word_data, rank, total_words, combo, config, fallback_count
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

    return assignments, OptimizationMetrics(
        total_cost=metrics["total_cost"],
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


def process_corpus_json(config: GeneratorConfig) -> None:
    """Process corpus and generate optimized chord assignments"""
    # Load configuration
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

    layout = create_keyboard_mapping(config)

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
            "firstLastUsage": metrics.first_last_usage,
            "fallbackAssignments": metrics.fallback_assignments,
            "averageChordLength": metrics.average_chord_length,
            "singleLetterWords": metrics.single_letter_words,
        },
        "chords": {
            word: (
                assignments[word] if word in assignments and assignments[word] else None
            )
            for word in words
        },
    }

    input_filename = Path(config.corpus_json_file).name
    output_filename = f"ChordsFor_{input_filename}"
    try:
        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        print(f"\nOutput written to {output_filename}")

        print("\nOptimization Metrics:")
        print(f"Total Weighted Cost: {metrics.total_cost:.4f}")
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
    loaded_config = GeneratorConfig.load_config(args.config)
    if args.corpus_file:
        loaded_config.corpus_json_file = args.corpus_file

    process_corpus_json(loaded_config)
