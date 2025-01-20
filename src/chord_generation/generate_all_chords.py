"""
Algorithm for generating and analyzing chords:
1. Generate a list of all possible chords based on configured key sets
2. Calculate standalone costs for each chord
3. Output both the raw chords and their calculated costs
"""

import json
import logging
from itertools import combinations
from math import inf
from pathlib import Path
from typing import Dict, List, Optional

from src.chord_generation.standalone_metrics import StandaloneMetricCalculator
from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.layout import load_input_device_layout
from src.common.shared_types import ChordData

logger = logging.getLogger(__name__)

# Define base output directories
BASE_OUTPUT_DIR = Path("data/output")
CHORDS_OUTPUT_DIR = BASE_OUTPUT_DIR / "chords_only"
CHORDCOSTS_OUTPUT_DIR = BASE_OUTPUT_DIR / "chords_with_costs"
INTERMEDIATE_OUTPUT_DIR = BASE_OUTPUT_DIR / "debug_intermediate"
LOGS_OUTPUT_DIR = BASE_OUTPUT_DIR / "logs"


def ensure_output_dirs():
    """Create output directories if they don't exist"""
    for directory in [
        CHORDS_OUTPUT_DIR,
        CHORDCOSTS_OUTPUT_DIR,
        INTERMEDIATE_OUTPUT_DIR,
        LOGS_OUTPUT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def calculate_total_chord_cost(metrics: Dict[str, float]) -> float:
    """Calculate total cost of a chord as product of its applied metrics and their weights

    Args:
        metrics: Dictionary of metric names to their costs

    Returns:
        Total combined cost
    """
    total = 1.0
    for weight in metrics.values():
        if weight < 0:
            return inf
        else:
            total *= weight
    return total


def calculate_chords_costs(
    chords: List[ChordData],
    calculator: StandaloneMetricCalculator,
    generator_config: GeneratorConfig,
    given_benchmark: Optional[Benchmark] = None,
) -> Dict[ChordData, Dict]:
    """Calculate costs for each chord in a list of chords

    Args:
        chords: List of chord strings
        calculator: Initialized metric calculator
        key_positions: Input device layout data
        config: Generator configuration
        given_benchmark: Benchmark instance

    Returns:
        Dictionary mapping chords to their metric costs
    """
    chord_cost_dict = {}
    for i, chord in enumerate(chords):
        if given_benchmark:
            given_benchmark.update_phase(items_processed=i)

        # Calculate metrics for current chord
        metrics = calculator.calculate(chord)

        # Convert enum keys to strings for JSON serialization
        serializable_metrics = {
            metric_type.name: cost for metric_type, cost in metrics.costs.items()
        }

        # Calculate total cost
        total_cost = calculate_total_chord_cost(serializable_metrics)

        # Build result dictionary based on debug settings
        result = {"chord": chord.letters, "total_cost": total_cost}

        # Include detailed metrics only if debug mode is enabled with cost details
        if generator_config.debug.enabled and generator_config.debug.print_cost_details:
            result["metrics"] = serializable_metrics

        chord_cost_dict[chord] = result

    return chord_cost_dict


def generate_chords_and_costs(
    generator_config: GeneratorConfig, given_benchmark: Optional[Benchmark] = None
) -> Dict:
    """Generate all possible chord combinations and calculate their costs"""

    if given_benchmark:
        # Initialization phase
        given_benchmark.start_phase(BenchmarkPhase.INITIALIZATION)

    # Initialize all required components
    ensure_output_dirs()
    key_positions = load_input_device_layout(generator_config.keylayout_file)
    available_keys = set(key_positions.keys())

    if (
        not 1
        <= generator_config.min_letter_count
        <= generator_config.max_letter_count
        <= len(available_keys)
    ):
        raise ValueError("Invalid min/max letter count configuration")

    calculator = StandaloneMetricCalculator(generator_config, given_benchmark)
    layout_name = generator_config.keylayout_file.stem

    if given_benchmark:
        # End initialization phase
        given_benchmark.update_phase(items_processed=len(available_keys))
        given_benchmark.end_phase()
        # Chord generation phase
        given_benchmark.start_phase(BenchmarkPhase.CHORD_GENERATION)

    all_chords = []
    length_details = {}

    for length in range(
        generator_config.min_letter_count, generator_config.max_letter_count + 1
    ):
        current_combinations = combinations(available_keys, length)
        current_chords = [
            ChordData(
                letters="".join(combo),
                keys=tuple(key_positions[k] for k in combo),  # Convert list to tuple
                length=length,
            )
            for combo in current_combinations
        ]
        all_chords.extend(current_chords)

        length_details[length] = {
            "combinations": len(current_chords),
        }

        if given_benchmark:
            # Update progress regularly
            given_benchmark.update_phase(items_processed=len(all_chords))

        if (
            generator_config.debug.enabled
            and generator_config.debug.save_intermediate_results
        ):
            intermediate_file = (
                INTERMEDIATE_OUTPUT_DIR
                / f"Intermediate_Chords_{layout_name}_length{length}.json"
            )
            intermediate_data = {
                "length": length,
                "chords": [chord.letters for chord in current_chords],
                "count": len(current_chords),
            }
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump(intermediate_data, f, indent=2)

    if given_benchmark:
        # End generation phase
        given_benchmark.end_phase()
        # Output phase
        given_benchmark.start_phase(BenchmarkPhase.WRITING_OUTPUT)

    # Write chords
    chords_filename = f"Chords_{layout_name}_{generator_config.min_letter_count}to{generator_config.max_letter_count}"
    chords_file = CHORDS_OUTPUT_DIR / f"{chords_filename}.json"
    chord_file_contents = {
        "name": chords_filename,
        "min_length": generator_config.min_letter_count,
        "max_length": generator_config.max_letter_count,
        "chords": [chord.letters for chord in all_chords],
    }
    with open(chords_file, "w", encoding="utf-8") as f:
        json.dump(chord_file_contents, f, indent=2)

    if given_benchmark:
        # End output phase
        given_benchmark.update_phase(items_processed=len(chord_file_contents))
        given_benchmark.end_phase()
        # Start cost calculation phase
        given_benchmark.start_phase(BenchmarkPhase.CHORD_COST_CALCULATION)

    # Calculate costs
    chords_with_costs = calculate_chords_costs(
        all_chords, calculator, generator_config, given_benchmark
    )

    if given_benchmark:
        given_benchmark.end_phase()
        given_benchmark.start_phase(BenchmarkPhase.WRITING_OUTPUT)

    # Write chords with costs
    chord_costs_filename = f"ChordCosts_{layout_name}_{generator_config.min_letter_count}to{generator_config.max_letter_count}"
    chord_costs_file = CHORDCOSTS_OUTPUT_DIR / f"{chord_costs_filename}.json"
    chord_costs_file_contents = {
        "name": chord_costs_filename,
        "min_length": generator_config.min_letter_count,
        "max_length": generator_config.max_letter_count,
        "chords": [
            {
                "chord": chord.letters,
                "costs": chord_costs,
            }
            for chord, chord_costs in chords_with_costs.items()
        ],
    }
    with open(chord_costs_file, "w", encoding="utf-8") as f:
        json.dump(chord_costs_file_contents, f, indent=2)

    if given_benchmark:
        given_benchmark.update_phase(items_processed=len(chords_with_costs))
        given_benchmark.end_phase()
        given_benchmark.start_phase(BenchmarkPhase.FINALIZATION)

    if given_benchmark:
        given_benchmark.update_phase(items_processed=2)

        # Get benchmark results before ending finalization
        benchmark_results = given_benchmark.get_results()
        given_benchmark.end_phase()

    return {
        "total_combinations": len(all_chords),
        "chords_file": str(chords_file),
        "metrics_file": str(chord_costs_filename),
        "min_length": generator_config.min_letter_count,
        "max_length": generator_config.max_letter_count,
        "benchmark": benchmark_results if given_benchmark else None,
    }


if __name__ == "__main__":
    config = GeneratorConfig.load_config()
    if config.benchmark.enabled:
        benchmark = Benchmark(config.benchmark)
        stats = generate_chords_and_costs(config, benchmark)
        benchmark.end_phase()
    else:
        stats = generate_chords_and_costs(config)
