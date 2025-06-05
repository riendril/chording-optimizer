"""
Chord generation module for creating all possible chords within specified constraints.

This module generates all valid chord combinations based on keyboard layout,
ensuring each finger can only press one key per chord, and calculates usage costs.
"""

import logging
import os
from typing import Dict, List, Tuple

from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.layout import load_keyboard_layout
from src.common.shared_types import ChordCollection, ChordData, KeyPosition, Finger

logger = logging.getLogger(__name__)


def calculate_chord_usage_cost(
    chord_keys: List[Tuple[KeyPosition, str]], layout_usage_cost: Dict[str, float]
) -> float:
    """Calculate the usage cost for a chord based on its keys.

    Args:
        chord_keys: List of (KeyPosition, character) tuples
        layout_usage_cost: Dictionary mapping characters to usage costs

    Returns:
        Total usage cost for the chord
    """
    total_cost = 1.0

    for _, char in chord_keys:
        # Get the usage cost for this character, default to 'unknown' if not found
        char_cost = layout_usage_cost.get(char, layout_usage_cost["unknown"])
        total_cost *= char_cost

    return total_cost




def generate_chord_combinations(
    key_positions: Dict[str, KeyPosition],
    layout_usage_cost: Dict[str, float],
    min_length: int,
    max_length: int,
) -> List[ChordData]:
    """Generate all valid chord combinations within the specified constraints.

    Uses a recursive algorithm that only generates valid chords by ensuring
    each finger is used at most once during generation.

    Args:
        key_positions: Dictionary mapping characters to KeyPosition objects
        layout_usage_cost: Dictionary mapping characters to usage costs
        min_length: Minimum chord length
        max_length: Maximum chord length

    Returns:
        List of ChordData objects representing all valid chords
    """
    all_chords = []

    # Group keys by finger for efficient generation
    keys_by_finger = {}
    for char, key_pos in key_positions.items():
        finger = key_pos.finger
        if finger not in keys_by_finger:
            keys_by_finger[finger] = []
        keys_by_finger[finger].append((key_pos, char))

    logger.info(f"Generating chords from {len(key_positions)} keys across {len(keys_by_finger)} fingers")

    # Generate chords for each length
    for chord_length in range(min_length, max_length + 1):
        logger.info(f"Generating {chord_length}-key chords...")
        
        chords_for_length = generate_valid_chords_recursive(
            keys_by_finger, chord_length, layout_usage_cost
        )
        
        all_chords.extend(chords_for_length)
        logger.info(f"Generated {len(chords_for_length)} valid {chord_length}-key chords")

    logger.info(f"Generated {len(all_chords)} total valid chords")
    return all_chords


def generate_valid_chords_recursive(
    keys_by_finger: Dict[Finger, List[Tuple[KeyPosition, str]]],
    target_length: int,
    layout_usage_cost: Dict[str, float],
) -> List[ChordData]:
    """Recursively generate valid chords of a specific length.

    Args:
        keys_by_finger: Dictionary mapping fingers to their available keys
        target_length: Target chord length
        layout_usage_cost: Dictionary mapping characters to usage costs

    Returns:
        List of valid ChordData objects of the specified length
    """
    all_chords = []
    available_fingers = list(keys_by_finger.keys())
    
    def build_chord_recursive(
        current_chord: List[Tuple[KeyPosition, str]],
        remaining_length: int,
        finger_index: int
    ):
        """Recursive helper to build chords.
        
        Args:
            current_chord: Current chord being built
            remaining_length: How many more keys we need to add
            finger_index: Index in available_fingers list to consider next
        """
        # Base case: we've built a complete chord
        if remaining_length == 0:
            # Calculate usage cost
            usage_cost = calculate_chord_usage_cost(current_chord, layout_usage_cost)
            
            # Create ChordData object
            chord = ChordData(
                keys=tuple(current_chord),
                key_count=len(current_chord),
                usage_cost=usage_cost,
            )
            all_chords.append(chord)
            return
        
        # If we don't have enough fingers left to complete the chord, backtrack
        if finger_index >= len(available_fingers):
            return
        
        # If we can't possibly reach target_length with remaining fingers, backtrack
        remaining_fingers = len(available_fingers) - finger_index
        if remaining_length > remaining_fingers:
            return
        
        current_finger = available_fingers[finger_index]
        
        # Option 1: Don't use this finger, move to next
        build_chord_recursive(current_chord, remaining_length, finger_index + 1)
        
        # Option 2: Use this finger (try each key on this finger)
        for key_pos, char in keys_by_finger[current_finger]:
            # Add this key to current chord
            current_chord.append((key_pos, char))
            
            # Recursively build the rest of the chord
            build_chord_recursive(current_chord, remaining_length - 1, finger_index + 1)
            
            # Backtrack: remove this key
            current_chord.pop()
    
    # Start the recursive generation
    build_chord_recursive([], target_length, 0)
    
    return all_chords


def generate_chord_collection(config: GeneratorConfig) -> ChordCollection:
    """Generate a complete chord collection based on configuration.

    Args:
        config: Generator configuration

    Returns:
        ChordCollection with all generated chords
    """
    # Load keyboard layout
    layout_path = config.paths.get_layout_path(config.active_layout_file)
    layout_data = load_keyboard_layout(layout_path)

    key_positions = layout_data["key_positions"]
    layout_usage_cost = layout_data["usage_cost"]

    logger.info(
        f"Loaded keyboard layout '{layout_data['name']}' with {len(key_positions)} keys"
    )

    # Generate all chord combinations
    chords = generate_chord_combinations(
        key_positions=key_positions,
        layout_usage_cost=layout_usage_cost,
        min_length=config.chord_generation.min_letter_count,
        max_length=config.chord_generation.max_letter_count,
    )

    # Create chord collection
    collection_name = f"{layout_data['name']}_chords_{config.chord_generation.min_letter_count}_{config.chord_generation.max_letter_count}"

    chord_collection = ChordCollection(
        name=collection_name,
        min_length=config.chord_generation.min_letter_count,
        max_length=config.chord_generation.max_letter_count,
        chords=sorted(chords, key=lambda chord: chord.usage_cost),
    )

    return chord_collection


def generate_chords(config: GeneratorConfig) -> None:
    """Generate chords and save them to file.

    Args:
        config: Generator configuration
    """
    # Initialize benchmarking if enabled
    benchmark = Benchmark(config.benchmark)

    logger.info("Starting chord generation")
    benchmark.start_phase(BenchmarkPhase.CHORD_GENERATION)

    # Generate chord collection
    chord_collection = generate_chord_collection(config)

    benchmark.update_phase(len(chord_collection.chords))
    benchmark.end_phase()

    # Create output filename
    layout_name = os.path.splitext(config.active_layout_file)[0]
    output_filename = f"{layout_name}_chords_{config.chord_generation.min_letter_count}_{config.chord_generation.max_letter_count}.json"
    output_path = config.paths.chords_dir / output_filename

    # Save to file
    logger.info(f"Saving {len(chord_collection.chords)} chords to {output_path}")
    benchmark.start_phase(BenchmarkPhase.WRITING_OUTPUT)
    chord_collection.save_to_file(output_path)
    benchmark.end_phase()

    # Log summary statistics
    logger.info(f"Generated chord collection '{chord_collection.name}':")
    logger.info(f"  Total chords: {len(chord_collection.chords)}")
    logger.info(
        f"  Length range: {chord_collection.min_length}-{chord_collection.max_length}"
    )

    # Log benchmark results if enabled
    if config.benchmark.enabled:
        results = benchmark.get_results()
        logger.info(f"Benchmark results: {results}")

    logger.info("Chord generation completed successfully")
