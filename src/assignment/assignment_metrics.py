"""
Assignment metrics for token-chord pair evaluation.

Usage:
    calculator = AssignmentMetricCalculator(config)
    token_data = TokenData(...)
    chord_data = ChordData(...)
    metrics = calculator.calculate(token_data, chord_data)
"""

from enum import Enum
from typing import Callable, Dict

from src.common.shared_types import (
    AssignmentMetricType,
    ChordData,
    SetMetricType,
    StandaloneCombinationType,
    StandaloneMetricType,
    TokenData,
)

AssignmentMetricFn = Callable[[TokenData, ChordData], float]


def calc_first_letter_unmatched(token: TokenData, chord: ChordData) -> float:
    """Calculate cost for first letter mismatch"""
    # Implementation
