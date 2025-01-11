"""
Algorithm for generating and analyzing chords:
1. Generate a list of all possible chords based on configured key sets
2. Calculate standalone metric costs for each chord
3. Output both the raw chords and their analyzed metrics
"""

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List

from src.chord_generation.standalone_metrics import StandaloneMetricCalculator
from src.common.benchmarking import Benchmark, BenchmarkPhase
from src.common.config import GeneratorConfig
from src.common.layout import KeyPosition, load_input_device_layout
from src.common.shared_types import ChordData

logger = logging.getLogger(__name__)

# Define base output directories
BASE_OUTPUT_DIR = Path("data/output")
CHORDS_OUTPUT_DIR = BASE_OUTPUT_DIR / "chordsOnly"
CHORDCOSTS_OUTPUT_DIR = BASE_OUTPUT_DIR / "chordsWithCosts"
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


def calculate_total_cost(metrics: Dict[str, float]) -> float:
    """Calculate total cost as product of individual metric costs

    Args:
        metrics: Dictionary of metric names to their costs

    Returns:
        Total combined cost
    """
    total = 1.0
    for cost in metrics.values():
        # Add 1 to each cost to prevent multiplication by zero
        # and to ensure costs < 1 don't reduce the total
        total *= 1.0 + cost
    return total


def analyze_chords(
    chords: List[str],
    calculator: StandaloneMetricCalculator,
    key_positions: Dict[str, KeyPosition],
    generator_config: GeneratorConfig,
    benchmark: Benchmark,
) -> Dict[str, Dict]:
    """Calculate metric costs for a list of chords

    Args:
        chords: List of chord strings to analyze
        calculator: Initialized metric calculator
        key_positions: Input device layout data
        config: Generator configuration
        benchmark: Benchmark manager instance

    Returns:
        Dictionary mapping chords to their metric costs
    """
    results = {}

    for i, chord in enumerate(chords):
        if (
            generator_config.benchmark.enabled
            and i % generator_config.benchmark.sample_interval == 0
        ):
            benchmark.update_phase(items_processed=i)

        # Calculate metrics for current chord
        chord_keys = [key_positions[c] for c in chord]
        chord_data = ChordData(length=len(chord), keys=chord_keys)
        metrics = calculator.calculate(chord_data)

        # Convert enum keys to strings for JSON serialization
        serializable_metrics = {
            metric_type.name: cost for metric_type, cost in metrics.costs.items()
        }

        # Calculate total cost
        total_cost = calculate_total_cost(serializable_metrics)

        # Build result dictionary based on debug settings
        result = {"chord": chord, "total_cost": total_cost}

        # Include detailed metrics only if debug mode is enabled with cost details
        if generator_config.debug.enabled and generator_config.debug.print_cost_details:
            result["metrics"] = serializable_metrics

        results[chord] = result

    return results


def generate_chords(generator_config: GeneratorConfig) -> Dict:
    """Generate all possible chord combinations and calculate their metrics"""
    benchmark = Benchmark(generator_config.benchmark)

    # Initialization phase
    benchmark.start_phase(BenchmarkPhase.INITIALIZATION)

    # Initialize all required components
    ensure_output_dirs()
    key_positions = load_input_device_layout(generator_config.keylayout_file)
    available_keys = set(key_positions.keys())

    if (
        not 1
        <= generator_config.min_letters
        <= generator_config.max_letters
        <= len(available_keys)
    ):
        raise ValueError("Invalid min/max letter configuration")

    calculator = StandaloneMetricCalculator(generator_config)
    base_name = generator_config.keylayout_file.stem

    # End initialization phase
    benchmark.update_phase(items_processed=len(available_keys))
    benchmark.end_phase()

    # Chord generation phase
    benchmark.start_phase(BenchmarkPhase.CHORD_GENERATION)
    all_chords = []
    length_details = {}

    for length in range(generator_config.min_letters, generator_config.max_letters + 1):
        current_combinations = combinations(available_keys, length)
        current_chords = ["".join(combo) for combo in current_combinations]
        all_chords.extend(current_chords)

        length_details[length] = {
            "combinations": len(current_chords),
        }

        # Update progress regularly
        benchmark.update_phase(items_processed=len(all_chords))

        if (
            generator_config.debug.enabled
            and generator_config.debug.save_intermediate_results
        ):
            intermediate_file = (
                INTERMEDIATE_OUTPUT_DIR
                / f"Intermediate_Chords_{base_name}_length{length}.json"
            )
            intermediate_data = {
                "length": length,
                "chords": current_chords,
                "count": len(current_chords),
            }
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump(intermediate_data, f, indent=2)

    base_filename = f"Chords_{base_name}_{generator_config.min_letters}to{generator_config.max_letters}"

    # Write output files
    output_data = {
        "name": base_filename,
        "min_length": generator_config.min_letters,
        "max_length": generator_config.max_letters,
        "chords": all_chords,
    }

    chords_file = CHORDS_OUTPUT_DIR / f"{base_filename}.json"
    with open(chords_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Calculate metrics
    metrics_file = (
        CHORDCOSTS_OUTPUT_DIR
        / f"ChordMetrics_{base_name}_{generator_config.min_letters}to{generator_config.max_letters}.json"
    )

    chord_metrics = analyze_chords(
        all_chords, calculator, key_positions, generator_config, benchmark
    )

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(chord_metrics, f, indent=2)

    benchmark.update_phase(items_processed=len(all_chords))
    benchmark.end_phase()

    # Get benchmark results
    results = benchmark.get_results()

    return {
        "total_combinations": len(all_chords),
        "chords_file": str(chords_file),
        "metrics_file": str(metrics_file),
        "min_length": generator_config.min_letters,
        "max_length": generator_config.max_letters,
        "benchmark": results if generator_config.benchmark.enabled else None,
    }


if __name__ == "__main__":
    config = GeneratorConfig.load_config()
    stats = generate_chords(config)
