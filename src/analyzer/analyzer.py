"""Analyzer module for evaluating chord and word assignments.

This module can analyze both types of assignment files:
1. Chord assignments (word -> chord mapping)
2. Word assignments (chord -> word mapping)

Usage from command line:
    python -m src.assignment_analyzing.analyzer path/to/assignment.json

The analyzer will automatically detect the assignment type and apply appropriate metrics.
Output will be saved to data/output/assignmentCosts/analysis_{filename}.json
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.common.config import GeneratorConfig
from src.common.layout import *
from src.common.shared_types import *


class ChordAnalyzer:
    """Analyzes chord assignments and calculates various metrics."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._load_weights()

    def _load_weights(self) -> None:
        """Load weights from configuration."""
        self.weights = {
            "vertical_stretch": self.config.vertical_stretch_weight,
            "vertical_pinch": self.config.vertical_pinch_weight,
            "horizontal_stretch": self.config.horizontal_stretch_weight,
            "horizontal_pinch": self.config.horizontal_pinch_weight,
            "diagonal_stretch": self.config.diagonal_stretch_weight,
            "diagonal_pinch": self.config.diagonal_pinch_weight,
            "same_finger_double": self.config.same_finger_double_weight,
            "same_finger_triple": self.config.same_finger_triple_weight,
            "pinky_ring_stretch": self.config.pinky_ring_stretch_weight,
            "ring_middle_scissor": self.config.ring_middle_scissor_weight,
            "middle_index_stretch": self.config.middle_index_stretch_weight,
            "first_letter_unmatched": self.config.first_letter_unmatched_weight,
            "second_letter_unmatched": self.config.second_letter_unmatched_weight,
            "last_letter_unmatched": self.config.last_letter_unmatched_weight,
            "additional_letter": self.config.additional_letter_weight,
            "fallback_letter": self.config.fallback_letter_weight,
        }

    def analyze_assignment_file(self, input_path: Path) -> Dict:
        """Analyze a chord or word assignment file and generate metrics.

        Args:
            input_path: Path to the assignment file. Can be either:
                - Chord assignment file (word -> chord mapping)
                - Word assignment file (chord -> word mapping)

        Returns:
            Dictionary containing the analysis results
        """
        with open(input_path, "r", encoding="utf-8") as f:
            assignments = json.load(f)

        if "chordAssignments" in assignments:
            is_chord_assignment = True
        elif "wordAssignments" in assignments:
            is_chord_assignment = False
        else:
            raise ValueError(
                "Invalid assignment file format - missing required assignment type"
            )
        metrics = self._calculate_metrics(assignments, is_chord_assignment)

        return {
            "name": assignments.get("name", "unnamed_analysis"),
            "input_file": input_path.name,
            "is_chord_assignment": is_chord_assignment,
            "metrics": metrics,
            "suggestions": self._generate_suggestions(assignments, metrics),
        }

    def _calculate_metrics(self, assignments: Dict, is_chord_assignment: bool) -> Dict:
        """Calculate all metrics for the assignment set.

        Args:
            assignments: Dictionary containing the assignments
            is_chord_assignment: If True, contains word->chord mappings
                               If False, contains chord->word mappings
        """
        assignment_dict = assignments.get(
            "chordAssignments" if is_chord_assignment else "wordAssignments", {}
        )

        # Calculate individual metrics
        chord_metrics = {}
        assignment_metrics = {}

        # Iterate through the assignments using more generic variable names
        for key, value in assignment_dict.items():
            if value is not None:
                if is_chord_assignment:
                    # key is word, value is chord
                    word, chord = key, value
                else:
                    # key is chord, value is word
                    chord, word = key, value

                chord_metrics[chord] = self._calculate_chord_metrics(chord)
                assignment_metrics[word] = self._calculate_assignment_metrics(
                    word, chord
                )

        # TODO: make work with word / chord assignments
        # Calculate global metrics
        global_metrics = self._calculate_global_metrics(assignment_dict)

        return {
            "chord_metrics": chord_metrics,
            "assignment_metrics": assignment_metrics,
            "global_metrics": global_metrics,
            "summary": self._calculate_summary_metrics(
                chord_metrics, assignment_metrics, global_metrics
            ),
        }

    def _generate_suggestions(self, assignments: Dict, metrics: Dict) -> List[Dict]:
        """Generate improvement suggestions based on metrics."""
        suggestions = []
        # Implementation would analyze metrics and generate suggestions
        return suggestions

    def save_analysis(self, analysis: Dict, output_path: Path) -> None:
        """Save analysis results to a JSON file.

        Args:
            analysis: Dictionary containing the analysis results
            output_path: Path where the JSON file should be saved
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)


def analyze_assignments(
    input_file: Path,
    config_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Analyze chord or word assignments and save results.

    Args:
        input_file: Path to the assignment file (either chord or word assignments)
        config_file: Optional path to config file (default: generator.config)
        output_dir: Optional output directory (default: data/output/assignmentCosts)
    """
    """Analyze chord assignments and save results."""
    # Load configuration
    config = GeneratorConfig.load_config(config_file)

    # Create analyzer
    analyzer = ChordAnalyzer(config)

    # Perform analysis
    analysis = analyzer.analyze_assignment_file(input_file)

    # Determine output path
    if output_dir is None:
        output_dir = Path("data/output/assignmentCosts")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"analysis_{input_file.stem}.json"

    # Save results
    analyzer.save_analysis(analysis, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze chord or word assignments and generate metrics"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the assignment file (either chord or word assignments)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: generator.config)",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: data/output/assignmentCosts)",
        default=None,
    )

    args = parser.parse_args()
    analyze_assignments(args.input_file, args.config, args.output_dir)
