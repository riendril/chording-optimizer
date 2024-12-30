"""
Assignment metrics for word-chord pair evaluation.

Usage:
    calculator = AssignmentMetricCalculator(config)
    word_data = WordData(...)
    chord_data = ChordData(...)
    metrics = calculator.calculate(word_data, chord_data)
"""

from enum import Enum
from typing import Callable, Dict

from src.common.shared_types import (
    AssignmentMetricType,
    ChordData,
    SetMetricType,
    StandaloneCombinationType,
    StandaloneMetricType,
    WordData,
)

AssignmentMetricFn = Callable[[WordData, ChordData], float]


def calc_first_letter_unmatched(word: WordData, chord: ChordData) -> float:
    """Calculate cost for first letter mismatch"""
    # Implementation
