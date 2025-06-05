"""Types shared across the chord generator modules"""

import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class Finger(Enum):
    """Enum for fingers that can be assigned to keys"""

    L_PINKY = auto()
    L_RING = auto()
    L_MIDDLE = auto()
    L_INDEX = auto()
    L_THUMB = auto()
    R_THUMB = auto()
    R_INDEX = auto()
    R_MIDDLE = auto()
    R_RING = auto()
    R_PINKY = auto()


class TokenType(IntEnum):
    """Types of tokens with increasing learning complexity"""

    SINGLE_CHARACTER = 0
    FULL_WORD = 1
    NGRAM_LETTERS_ONLY = 2
    WORD_FOLLOWED_BY_SPACE = 3
    NGRAM_NO_LETTERS = 4
    OTHER = 5


@dataclass
class KeyPosition:
    """Position and finger assignment of a key on the input device"""

    finger: "Finger"
    vertical_distance_to_resting_position: int
    horizontal_distance_to_resting_position: int
    finger_to_left: Optional["Finger"]
    finger_to_right: Optional["Finger"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "finger": self.finger.name,
            "vertical_distance": self.vertical_distance_to_resting_position,
            "horizontal_distance": self.horizontal_distance_to_resting_position,
            "finger_to_left": self.finger_to_left.name if self.finger_to_left else None,
            "finger_to_right": (
                self.finger_to_right.name if self.finger_to_right else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyPosition":
        """Create from dictionary after JSON deserialization"""
        return cls(
            finger=Finger[data["finger"]],
            vertical_distance_to_resting_position=data["vertical_distance"],
            horizontal_distance_to_resting_position=data["horizontal_distance"],
            finger_to_left=Finger[data["finger_to_left"]] if data["finger_to_left"] is not None else None,
            finger_to_right=Finger[data["finger_to_right"]] if data["finger_to_right"] is not None else None,
        )


@dataclass
class TokenData:
    """Represents preprocessed data for a token"""

    lower: str
    character_length: int
    subtoken_length: int
    token_type: TokenType
    text_count: int
    usage_count: int
    rank: int
    usage_cost: float
    replacement_score: float
    selected: bool
    best_current_combination: List[str]
    adjacent_tokens: Optional[Dict[str, Dict[str, int]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "lower": self.lower,
            "character_length": self.character_length,
            "subtoken_length": self.subtoken_length,
            "token_type": self.token_type.name,
            "text_count": self.text_count,
            "usage_count": self.usage_count,
            "rank": self.rank,
            "usage_cost": self.usage_cost,
            "replacement_score": self.replacement_score,
            "selected": self.selected,
            "best_current_combination": self.best_current_combination,
            "adjacent_tokens": self.adjacent_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenData":
        """Create from dictionary after JSON deserialization"""
        return cls(
            lower=data["lower"],
            character_length=data["character_length"],
            subtoken_length=data["subtoken_length"],
            token_type=TokenType[data["token_type"]],
            text_count=data["text_count"],
            usage_count=data["usage_count"],
            rank=data["rank"],
            usage_cost=data["usage_cost"],
            replacement_score=data["replacement_score"],
            selected=data["selected"],
            best_current_combination=data["best_current_combination"],
            adjacent_tokens=data["adjacent_tokens"],
        )


@dataclass
class TokenCollection:
    """Collection of tokens with frequency information"""

    name: str
    tokens: List[TokenData]
    ordered_by_frequency: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "tokens": [token.to_dict() for token in self.tokens],
            "orderedByFrequency": self.ordered_by_frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenCollection":
        """Create from dictionary after JSON deserialization"""
        return cls(
            name=data["name"],
            tokens=[TokenData.from_dict(token_data) for token_data in data["tokens"]],
            ordered_by_frequency=data["orderedByFrequency"],
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save token collection to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "TokenCollection":
        """Load token collection from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ChordData:
    """Represents preprocessed data for a chord"""

    keys: Tuple[Tuple[KeyPosition, str]]
    key_count: int
    usage_cost: float

    def __post_init__(self):
        self.key_count = len(self.keys)

    @property
    def letters(self) -> str:
        """Get the letters from the chord keys in order"""
        return "".join(char for _, char in self.keys)

    @property
    def character_length(self) -> int:
        """Get the character length (for backward compatibility)"""
        return self.key_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "keys": [
                {"key_position": key_pos.to_dict(), "character": char}
                for key_pos, char in self.keys
            ],
            "key_count": self.key_count,
            "usage_cost": self.usage_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChordData":
        """Create from dictionary after JSON deserialization"""
        return cls(
            keys=tuple(
                (
                    KeyPosition.from_dict(key_tuple["key_position"]),
                    key_tuple["character"],
                )
                for key_tuple in data["keys"]
            ),
            key_count=data["key_count"],
            usage_cost=data["usage_cost"],
        )


@dataclass
class ChordCollection:
    """Collection of chords with their properties"""

    name: str
    min_length: int
    max_length: int
    chords: List[ChordData]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "chords": [chord.to_dict() for chord in self.chords],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChordCollection":
        """Create from dictionary after JSON deserialization"""
        return cls(
            name=data["name"],
            min_length=data["min_length"],
            max_length=data["max_length"],
            chords=[ChordData.from_dict(chord_data) for chord_data in data["chords"]],
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save chord collection to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "ChordCollection":
        """Load chord collection from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class Assignment:
    """Represents a token-chord assignment pair"""

    token: TokenData
    chord: ChordData
    metrics: Optional[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "token": self.token.to_dict(),
            "chord": self.chord.to_dict(),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Assignment":
        """Create from dictionary after JSON deserialization"""
        return cls(
            token=TokenData.from_dict(data["token"]),
            chord=ChordData.from_dict(data["chord"]),
            metrics=data["metrics"],
        )


@dataclass
class AssignmentSet:
    """Represents a set of assignments between tokens and chords"""

    name: str
    assignments: List[Assignment]
    metrics: Optional[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssignmentSet":
        """Create from dictionary after JSON deserialization"""
        return cls(
            name=data["name"],
            assignments=[
                Assignment.from_dict(assignment_data)
                for assignment_data in data["assignments"]
            ],
            metrics=data["metrics"],
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save assignment set to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "AssignmentSet":
        """Load assignment set from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class SetData:
    """Data for evaluating sets of assignments"""

    assignment_set: AssignmentSet
    chord_collection: ChordCollection
    token_collection: TokenCollection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "assignment_set": self.assignment_set.to_dict(),
            "chord_collection": self.chord_collection.to_dict(),
            "token_collection": self.token_collection.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SetData":
        """Create from dictionary after JSON deserialization"""
        return cls(
            assignment_set=AssignmentSet.from_dict(data["assignment_set"]),
            chord_collection=ChordCollection.from_dict(data["chord_collection"]),
            token_collection=TokenCollection.from_dict(data["token_collection"]),
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save SetData to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "SetData":
        """Load SetData from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class StandaloneMetricType(Enum):
    """Types of metrics for standalone chord analysis"""

    CHORD_LENGTH = auto()
    HORIZONTAL_PINCH = auto()
    HORIZONTAL_STRETCH = auto()
    VERTICAL_PINCH = auto()
    VERTICAL_STRETCH = auto()
    DIAGONAL_DOWNWARD_PINCH = auto()
    DIAGONAL_DOWNWARD_STRETCH = auto()
    DIAGONAL_UPWARD_PINCH = auto()
    DIAGONAL_UPWARD_STRETCH = auto()
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


class StandaloneCombinationType(Enum):
    """How to combine individual standalone metrics"""

    WEIGHTED_SUM = auto()
    MAX = auto()
    PRODUCT = auto()


class AssignmentMetricType(Enum):
    """Types of metrics for token-chord assignments"""

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
