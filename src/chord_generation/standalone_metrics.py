"""
Standalone metrics for individual chord evaluation.

Usage:
    calculator = StandaloneMetricCalculator(config)
    chord_data = ChordData(...)
    metrics = calculator.calculate(chord_data)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict

from src.common.shared_types import (
    AssignmentMetricType,
    SetMetricType,
    StandaloneCombinationType,
    StandaloneMetricType,
)

# Type alias for standalone metric calculation functions
StandaloneMetricFn = Callable[[ChordData], float]


def calc_horizontal_stretch(chord: ChordData) -> float:
    """Calculate horizontal stretch cost for a chord.

    Args:
        chord: Preprocessed chord data including letter positions

    Returns:
        Cost value where higher means more stretch
    """
    # Implementation


class StandaloneMetricCalculator:
    """Calculates metrics for individual chords using configured weights"""

    def __init__(self, config: GeneratorConfig):
        self.weights = config.standalone_weights

    def calculate(self, chord: ChordData) -> StandaloneMetrics:
        """Calculate all metrics for a single chord

        Args:
            chord: Preprocessed chord data

        Returns:
            Complete set of weighted metric costs for the chord
        """
        movement_costs = {
            metric_type: self.weights.movement_weights[metric_type]
            * metric_type.value(chord)
            for metric_type in StandaloneMetricType
        }
        # Similar for combination costs
        return StandaloneMetrics(...)
