"""
Set metrics for evaluating collections of assignments.

Usage:
    calculator = SetMetricCalculator(config)
    set_data = SetData(...)
    metrics = calculator.calculate(set_data)
"""

from enum import Enum
from typing import Callable, Dict

from src.common.shared_types import SetData

SetMetricFn = Callable[[SetData], float]


class SetMetricType(Enum):
    """Metrics for analyzing sets of assignments"""

    FINGER_UTILIZATION = "finger_utilization"
    # ... other types


def calc_finger_utilization(set_data: SetData) -> float:
    """Calculate finger utilization balance cost"""
    # Implementation
