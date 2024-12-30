from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List

from src.common.layout import LetterData


@dataclass
class WordData:
    """Represents preprocessed data for a word"""

    original: str
    lower: str
    length: int
    zipf_weight: float


@dataclass
class ChordData:
    """Represents preprocessed data for a chord"""

    length: int
    letters: List[LetterData]


@dataclass
class SetData:
    """Represents preprocessed data for a set of assignments"""

    length: int
    assignments: Dict[WordData, ChordData]


class StandaloneMetricType(Enum):
    """Types of metrics for standalone chord analysis"""

    CHORD_LENGTH = auto()
    HORIZONTAL_STRETCH = auto()
    VERTICAL_STRETCH = auto()
    DIAGONAL_STRETCH = auto()
    HORIZONTAL_PINCH = auto()
    VERTICAL_PINCH = auto()
    DIAGONAL_PINCH = auto()
    SAME_FINGER_DOUBLE_ADJACENT = auto()
    SAME_FINGER_DOUBLE_GAP = auto()
    SAME_FINGER_TRIPLE = auto()
    FULL_SCISSOR_DOUBLE = auto()
    FULL_SCISSOR_TRIPLE = auto()
    FULL_SCISSOR_QUADRUPLE = auto()
    FULL_SCISSOR_QUINTUPLE = auto()
    HALF_SCISSOR_DOUBLE = auto()
    HORIZONTAL_STRETCH_DOUBLE = auto()
    PINKY_RING_SCISSOR = auto()
    RING_INDEX_SCISSOR = auto()


class AssignmentMetricType(Enum):
    """Types of metrics for word-chord assignments"""

    FIRST_LETTER_UNMATCHED = auto()
    SECOND_LETTER_UNMATCHED = auto()
    LAST_LETTER_UNMATCHED = auto()
    PHONETIC_DISSIMILARITY = auto()
    EXTRA_LETTER = auto()


class SetMetricType(Enum):
    """Types of metrics for sets of assignments"""

    FINGER_UTILIZATION = auto()
    HAND_UTILIZATION = auto()
    CHORD_PATTERN_CONSISTENCY = auto()


@dataclass
class StandaloneMetrics:
    """Metrics for evaluating individual chords"""

    costs: Dict[StandaloneMetricType, float]


@dataclass
class AssignmentMetrics:
    """Metrics for evaluating word-chord assignments"""

    costs: Dict[AssignmentMetricType, float]
    word_frequency: float  # From Zipf's law


@dataclass
class SetMetrics:
    """Metrics for evaluating sets of assignments"""

    costs: Dict[SetMetricType, float]
