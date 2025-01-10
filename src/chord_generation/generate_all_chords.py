"""
Algorithm for generating and analyzing chords:
1. Generate a list of all possible chords based on configured key sets
2. Calculate standalone metric costs for each chord
3. Output both the raw chords and their analyzed metrics

Usage:
    stats = generate_chords(config)
    
The generator produces two files in data/output:
- chordsOnly/Chords_{layout}_{keyset}_{min}to{max}.json: Raw chord combinations
- chordsWithCosts/ChordMetrics_{layout}_{keyset}_{min}to{max}.json: Chords with their metric costs
"""

import json
import logging
import os
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import psutil
import tabulate
from standalone_metrics import StandaloneMetricCalculator

from src.common.config import GeneratorConfig
from src.common.layout import KeyPosition, load_input_device_layout
from src.common.shared_types import ChordData

logger = logging.getLogger(__name__)

# Define base output directories
BASE_OUTPUT_DIR = Path("data/output")
CHORDS_OUTPUT_DIR = BASE_OUTPUT_DIR / "chordsOnly"
CHORDCOSTS_OUTPUT_DIR = BASE_OUTPUT_DIR / "chordsWithCosts"
INTERMEDIATE_OUTPUT_DIR = BASE_OUTPUT_DIR / "debug_intermediate"


def ensure_output_dirs():
    """Create output directories if they don't exist"""
    for directory in [
        CHORDS_OUTPUT_DIR,
        CHORDCOSTS_OUTPUT_DIR,
        INTERMEDIATE_OUTPUT_DIR,
        LOGS_OUTPUT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def format_benchmark_results(benchmark_data: Dict) -> None:
    """Format benchmark results into a nice table format"""
    # Prepare header
    print("\nBenchmark Results:")
    print(f"Timestamp: {benchmark_data['timestamp']}")
    print(f"Total Time: {benchmark_data['total_time']:.4f} seconds")
    print(f"Total Memory Change: {benchmark_data['total_memory']:.2f} MB\n")

    # Prepare data for length details table
    headers = ["Length", "Combinations", "Time (s)", "Memory (MB)"]
    table_data = []
    cumulative_combinations = 0

    for length, details in benchmark_data["length_details"].items():
        cumulative_combinations += details["combinations"]
        table_data.append(
            [
                length,
                f"{details['combinations']:,}",
                f"{details['time']:.4f}",
                f"{details['memory_change']:.2f}",
            ]
        )

    # Add total row
    table_data.append(
        [
            "Total",
            f"{cumulative_combinations:,}",
            f"{benchmark_data['total_time']:.4f}",
            f"{benchmark_data['total_memory']:.2f}",
        ]
    )

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()


def analyze_chords(
    chords: List[str],
    calculator: StandaloneMetricCalculator,
    key_positions: Dict[str, KeyPosition],
    config: GeneratorConfig,
) -> Dict[str, Dict]:
    """Calculate metric costs for a list of chords

    Args:
        chords: List of chord strings to analyze
        calculator: Initialized metric calculator
        layout_data: Input device layout data
        config: Generator configuration

    Returns:
        Dictionary mapping chords to their metric costs
    """
    results = {}
    total = len(chords)

    for i, chord in enumerate(chords):
        if (
            config.benchmark.enabled
            and config.benchmark.print_progress
            and i % 1000 == 0
        ):
            logger.info(f"Analyzing chord {i}/{total}")

        chord_keys = [key_positions[c] for c in chord]
        chord_data = ChordData(length=len(chord), keys=chord_keys)
        metrics = calculator.calculate(chord_data)

        # Convert enum keys to strings for JSON serialization
        serializable_metrics = {
            metric_type.name: cost for metric_type, cost in metrics.costs.items()
        }

        results[chord] = {"chord": chord, "metrics": serializable_metrics}

    return results


def generate_chords(config: GeneratorConfig) -> Dict:
    """Generate all possible chord combinations and calculate their metrics

    Args:
        config: Generator configuration containing layout and key set information

    Returns:
        Dictionary containing generation statistics
    """
    # Update layout initialization
    key_positions = load_input_device_layout(config.keylayout_file)

    # Get available keys directly from layout
    available_keys = set(key_positions.keys())

    if not 1 <= config.min_letters <= config.max_letters <= len(available_keys):
        raise ValueError("Invalid min/max letter configuration")

    # Ensure output directories exist
    ensure_output_dirs()

    # Initialize debug/benchmark if enabled
    if config.debug.enabled:
        config.setup_logging()

    start_time = time.time()
    start_memory = get_memory_usage() if config.benchmark.enabled else 0

    # Initialize metric calculator
    calculator = StandaloneMetricCalculator(config)

    length_timing = {}
    all_chords = []
    benchmark_data = {}

    # Get base name without extension
    base_name = config.keylayout_file.stem

    # Generate combinations of different lengths
    for length in range(config.min_letters, config.max_letters + 1):
        if config.benchmark.enabled:
            length_start_time = time.time()
            length_start_memory = get_memory_usage()

        current_combinations = combinations(available_keys, length)
        current_chords = ["".join(combo) for combo in current_combinations]
        all_chords.extend(current_chords)

        if config.debug.enabled and config.debug.save_intermediate_results:
            # Save intermediate results with proper path
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
            logger.info(
                f"Saved intermediate results for length {length} to {intermediate_file}"
            )

        if config.benchmark.enabled:
            length_timing[length] = {
                "combinations": len(current_chords),
                "time": time.time() - length_start_time,
                "memory_change": get_memory_usage() - length_start_memory,
            }

    # Create the base output data
    base_filename = f"Chords_{base_name}_{config.min_letters}to{config.max_letters}"
    output_data = {
        "name": base_filename,
        "min_length": config.min_letters,
        "max_length": config.max_letters,
        "chords": all_chords,
    }

    # Handle benchmarking data
    if config.benchmark.enabled:
        benchmark_data = {
            "timestamp": datetime.now().isoformat(),
            "total_time": time.time() - start_time,
            "total_memory": get_memory_usage() - start_memory,
            "length_details": length_timing,
        }

        if config.benchmark.output_file:
            with open(config.benchmark.output_file, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2)
            logger.info(f"Saved benchmark data to {config.benchmark.output_file}")
        else:
            output_data["benchmark"] = benchmark_data

        if config.benchmark.print_progress:
            format_benchmark_results(benchmark_data)

    # Write raw chords to file in the chords directory
    chords_file = CHORDS_OUTPUT_DIR / f"{base_filename}.json"
    with open(chords_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Calculate and write metric costs
    metrics_file = (
        CHORDCOSTS_OUTPUT_DIR
        / f"ChordMetrics_{base_name}_{config.min_letters}to{config.max_letters}.json"
    )
    chord_metrics = analyze_chords(all_chords, calculator, key_positions, config)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(chord_metrics, f, indent=2)

    # Return statistics
    stats = {
        "total_combinations": len(all_chords),
        "chords_file": str(chords_file),
        "metrics_file": str(metrics_file),
        "min_length": config.min_letters,
        "max_length": config.max_letters,
    }

    if config.benchmark.enabled:
        stats["benchmark"] = benchmark_data

    return stats


# Example usage
if __name__ == "__main__":
    config = GeneratorConfig.load_config()
    stats = generate_chords(config)
