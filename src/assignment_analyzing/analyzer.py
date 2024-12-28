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

from src.get_parameters import GeneratorConfig


@dataclass
class ChordMetrics:
    """Metrics for individual chord evaluation."""

    length: int
    home_row_deviation: float
    stretch_pinch_score: float
    movement_combinations_score: float


@dataclass
class AssignmentMetrics:
    """Metrics for word-chord assignment evaluation."""

    visual_similarity: float
    phonetic_similarity: float
    missing_letters_score: float
    extra_letters_score: float
    ngram_preservation: float
    word_priority: float


@dataclass
class GlobalMetrics:
    """Global metrics for entire assignment set."""

    finger_utilization: Dict[str, float]
    hand_balance: float
    chord_pattern_consistency: float
    sequence_difficulty: float


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

        is_chord_assignment = "chordAssignments" in assignments
        metrics = self._calculate_metrics(assignments, is_chord_assignment)

        return {
            "name": assignments.get("name", "unnamed_analysis"),
            "input_file": input_path.name,
            "is_chord_assignment": is_chord_assignment,
            "metrics": metrics,
            "suggestions": self._generate_suggestions(assignments, metrics),
        }

    def _calculate_metrics(self, assignments: Dict, is_chord_assignment: bool) -> Dict:
        """Calculate all metrics for the assignment set."""
        assignment_dict = assignments.get(
            "chordAssignments" if is_chord_assignment else "wordAssignments", {}
        )

        # Calculate individual metrics
        chord_metrics = {}
        assignment_metrics = {}
        for word, chord in assignment_dict.items():
            if chord is not None:
                chord_metrics[chord] = self._calculate_chord_metrics(chord)
                assignment_metrics[word] = self._calculate_assignment_metrics(
                    word, chord
                )

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

    def _calculate_chord_metrics(self, chord: str) -> ChordMetrics:
        """Calculate metrics for a single chord."""
        return ChordMetrics(
            length=len(chord),
            home_row_deviation=self._calculate_home_row_deviation(chord),
            stretch_pinch_score=self._calculate_stretch_pinch_score(chord),
            movement_combinations_score=self._calculate_movement_combinations(chord),
        )

    def _calculate_assignment_metrics(self, word: str, chord: str) -> AssignmentMetrics:
        """Calculate metrics for a word-chord assignment."""
        return AssignmentMetrics(
            visual_similarity=self._calculate_visual_similarity(word, chord),
            phonetic_similarity=self._calculate_phonetic_similarity(word, chord),
            missing_letters_score=self._calculate_missing_letters_score(word, chord),
            extra_letters_score=self._calculate_extra_letters_score(word, chord),
            ngram_preservation=self._calculate_ngram_preservation(word, chord),
            word_priority=self._calculate_word_priority(word),
        )

    def _calculate_global_metrics(self, assignments: Dict[str, str]) -> GlobalMetrics:
        """Calculate global metrics for the entire assignment set."""
        return GlobalMetrics(
            finger_utilization=self._calculate_finger_utilization(assignments),
            hand_balance=self._calculate_hand_balance(assignments),
            chord_pattern_consistency=self._calculate_chord_pattern_consistency(
                assignments
            ),
            sequence_difficulty=self._calculate_sequence_difficulty(assignments),
        )

    def _calculate_home_row_deviation(self, chord: str) -> float:
        """Calculate deviation from home row position."""
        # Implementation would depend on your keyboard layout
        return 0.0  # Placeholder

    def _calculate_stretch_pinch_score(self, chord: str) -> float:
        """Calculate score for stretches and pinches in chord."""
        # Implementation would depend on your keyboard layout
        return 0.0  # Placeholder

    def _calculate_movement_combinations(self, chord: str) -> float:
        """Calculate score for movement combinations in chord."""
        # Implementation would depend on your keyboard layout
        return 0.0  # Placeholder

    def _calculate_visual_similarity(self, word: str, chord: str) -> float:
        """Calculate visual similarity between word and chord."""
        # Implementation would use string similarity metrics
        return 0.0  # Placeholder

    def _calculate_phonetic_similarity(self, word: str, chord: str) -> float:
        """Calculate phonetic similarity between word and chord."""
        # Implementation would use phonetic algorithms
        return 0.0  # Placeholder

    def _calculate_missing_letters_score(self, word: str, chord: str) -> float:
        """Calculate score for letters missing from word in chord."""
        word_set = set(word.lower())
        chord_set = set(chord.lower())
        missing_letters = word_set - chord_set

        score = 0.0
        for letter in missing_letters:
            if letter == word[0]:
                score += self.weights["first_letter_unmatched"]
            elif len(word) > 1 and letter == word[1]:
                score += self.weights["second_letter_unmatched"]
            elif letter == word[-1]:
                score += self.weights["last_letter_unmatched"]

        return score

    def _calculate_extra_letters_score(self, word: str, chord: str) -> float:
        """Calculate score for extra letters in chord not in word."""
        return (
            len(set(chord.lower()) - set(word.lower()))
            * self.weights["additional_letter"]
        )

    def _calculate_ngram_preservation(self, word: str, chord: str) -> float:
        """Calculate how well the chord preserves word n-grams."""
        # Implementation would analyze bigrams and trigrams
        return 0.0  # Placeholder

    def _calculate_word_priority(self, word: str) -> float:
        """Calculate priority score for word."""
        # Implementation would consider word frequency and difficulty
        return 0.0  # Placeholder

    def _calculate_finger_utilization(
        self, assignments: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate finger utilization statistics."""
        # Implementation would track finger usage
        return {}  # Placeholder

    def _calculate_hand_balance(self, assignments: Dict[str, str]) -> float:
        """Calculate hand balance score."""
        # Implementation would analyze left/right hand usage
        return 0.0  # Placeholder

    def _calculate_chord_pattern_consistency(
        self, assignments: Dict[str, str]
    ) -> float:
        """Calculate consistency of chord patterns."""
        # Implementation would analyze pattern similarities
        return 0.0  # Placeholder

    def _calculate_sequence_difficulty(self, assignments: Dict[str, str]) -> float:
        """Calculate difficulty of chord sequences."""
        # Implementation would analyze transitions between chords
        return 0.0  # Placeholder

    def _calculate_summary_metrics(
        self,
        chord_metrics: Dict[str, ChordMetrics],
        assignment_metrics: Dict[str, AssignmentMetrics],
        global_metrics: GlobalMetrics,
    ) -> Dict:
        """Calculate summary metrics for the entire analysis."""
        return {
            "average_chord_length": sum(m.length for m in chord_metrics.values())
            / len(chord_metrics),
            "average_movement_score": sum(
                m.movement_combinations_score for m in chord_metrics.values()
            )
            / len(chord_metrics),
            "hand_balance": global_metrics.hand_balance,
            "pattern_consistency": global_metrics.chord_pattern_consistency,
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
