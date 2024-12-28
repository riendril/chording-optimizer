"""
Algorithm 2:
1. Generate a list of all chords
2. Order by their score
3. Try assigning words to chords
"""

import json
import os
import string
import time
from datetime import datetime
from itertools import combinations

import psutil
from tabulate import tabulate


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def format_benchmark_results(benchmark_data):
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


def generate_english_chords(n, benchmark=False, print_results=True):
    """
    Generate all possible combinations of English letters from length 1 to n and save to JSON.

    Args:
        n (int): Maximum length of combinations (1 <= n <= 26)
        benchmark (bool): Whether to include benchmarking information
        print_results (bool): Whether to print formatted benchmark results

    Returns:
        dict: Statistics about the generation process
    """
    if not 1 <= n <= 26:
        raise ValueError("n must be between 1 and 26 inclusive")

    # Initialize benchmarking metrics
    start_time = time.time()
    start_memory = get_memory_usage()

    # Get lowercase English alphabet
    letters = string.ascii_lowercase

    # Initialize list to store all combinations
    all_chords = []

    # Dictionary to store timing for each length
    length_timing = {}

    # Generate combinations of different lengths
    for length in range(1, n + 1):
        length_start_time = time.time()
        length_start_memory = get_memory_usage()

        # Generate all combinations of current length
        current_combinations = combinations(letters, length)
        # Convert each combination tuple to a sorted string and add to list
        current_chords = ["".join(combo) for combo in current_combinations]
        all_chords.extend(current_chords)

        if benchmark:
            length_timing[length] = {
                "combinations": len(current_chords),
                "time": round(time.time() - length_start_time, 4),
                "memory_change": round(get_memory_usage() - length_start_memory, 2),
            }

    # Create the output dictionary
    output_data = {"name": f"english_{n}", "order": str(n), "chords": all_chords}

    # Add benchmarking data if requested
    if benchmark:
        benchmark_data = {
            "timestamp": datetime.now().isoformat(),
            "total_time": round(time.time() - start_time, 4),
            "total_memory": round(get_memory_usage() - start_memory, 2),
            "length_details": length_timing,
        }
        output_data["benchmark"] = benchmark_data

        if print_results:
            format_benchmark_results(benchmark_data)

    # Write to JSON file
    filename = f"Chords_english_{n}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Return statistics
    stats = {
        "total_combinations": len(all_chords),
        "file_created": filename,
        "max_length": n,
    }

    if benchmark:
        stats["benchmark"] = benchmark_data

    return stats


# Example usage
if __name__ == "__main__":
    # Generate combinations up to length 4 with benchmarking
    stats = generate_english_chords(12, benchmark=True)
