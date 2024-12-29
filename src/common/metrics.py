from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict


class StandaloneMovementType(Enum):
    """Types of movements for standalone chord analysis"""

    HORIZONTAL_STRETCH = auto()
    VERTICAL_STRETCH = auto()
    DIAGONAL_STRETCH = auto()
    HORIZONTAL_PINCH = auto()
    VERTICAL_PINCH = auto()
    DIAGONAL_PINCH = auto()


class StandaloneCombinationType(Enum):
    """Types of finger combinations for standalone chord analysis"""

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

    length: int
    movement_costs: Dict[StandaloneMovementType, float]
    combination_costs: Dict[StandaloneCombinationType, float]


@dataclass
class AssignmentMetrics:
    """Metrics for evaluating word-chord assignments"""

    unmatched_costs: Dict[AssignmentMetricType, float]
    word_frequency: float  # From Zipf's law


@dataclass
class SetMetrics:
    """Metrics for evaluating sets of assignments"""

    costs: Dict[SetMetricType, float]


# TODO: make use of different data structures
# TODO: Change calculation
def calculate_chord_metrics(self, chord: str) -> ChordMetrics:
    """Calculate metrics for a single chord."""
    return ChordMetrics(
        length=len(chord),
        home_row_deviation=self._calculate_home_row_deviation(chord),
        stretch_pinch_score=self._calculate_stretch_pinch_score(chord),
        movement_combinations_score=self._calculate_movement_combinations(chord),
    )


def calculate_assignment_metrics(self, word: str, chord: str) -> AssignmentMetrics:
    """Calculate metrics for a word-chord assignment."""
    return AssignmentMetrics(
        visual_similarity=self._calculate_visual_similarity(word, chord),
        phonetic_similarity=self._calculate_phonetic_similarity(word, chord),
        missing_letters_score=self._calculate_missing_letters_score(word, chord),
        extra_letters_score=self._calculate_extra_letters_score(word, chord),
        ngram_preservation=self._calculate_ngram_preservation(word, chord),
        word_priority=self._calculate_word_priority(word),
    )


def calculate_global_metrics(self, assignments: Dict[str, str]) -> GlobalMetrics:
    """Calculate global metrics for the entire assignment set."""
    return GlobalMetrics(
        finger_utilization=self._calculate_finger_utilization(assignments),
        hand_balance=self._calculate_hand_balance(assignments),
        chord_pattern_consistency=self._calculate_chord_pattern_consistency(
            assignments
        ),
        sequence_difficulty=self._calculate_sequence_difficulty(assignments),
    )


def calculate_home_row_deviation(self, chord: str) -> float:
    """Calculate deviation from home row position."""
    if not hasattr(self, "keyboard_layout"):
        self.keyboard_layout = self._load_keyboard_layout(
            Path(self.config.keylayout_csv_file)
        )

    total_deviation = 0.0
    for letter in chord:
        if letter in self.keyboard_layout:
            letter_data = self.keyboard_layout[letter]
            total_deviation += abs(letter_data.vertical_distance_to_home_row)
            total_deviation += abs(letter_data.horizontal_distance_to_home_row)

    return total_deviation / len(chord) if chord else 0.0


def calculate_stretch_pinch_score(self, chord: str) -> float:
    """Calculate score for stretches and pinches in chord."""
    if not hasattr(self, "keyboard_layout"):
        self.keyboard_layout = self._load_keyboard_layout(
            Path(self.config.keylayout_csv_file)
        )

    score = 0.0
    chord_letters = [
        self.keyboard_layout[c] for c in chord if c in self.keyboard_layout
    ]

    for i, curr in enumerate(chord_letters):
        # Vertical stretches and pinches
        if curr.vertical_distance_to_home_row != 0:
            if abs(curr.vertical_distance_to_home_row) > 1:
                score += self.weights["vertical_stretch"]
            for prev in chord_letters[:i]:
                if prev.vertical_distance_to_home_row != 0:
                    if (
                        prev.vertical_distance_to_home_row
                        * curr.vertical_distance_to_home_row
                    ) < 0:
                        score += self.weights["vertical_pinch"]

        # Horizontal stretches and pinches
        if curr.horizontal_distance_to_home_row != 0:
            if curr.finger_to_left in {l.finger for l in chord_letters[:i]}:
                score += self.weights["horizontal_stretch"]
            if curr.finger_to_right in {l.finger for l in chord_letters[:i]}:
                score += self.weights["horizontal_stretch"]
            for prev in chord_letters[:i]:
                if (
                    prev.horizontal_distance_to_home_row
                    * curr.horizontal_distance_to_home_row
                ) < 0:
                    score += self.weights["horizontal_pinch"]

    return score


def calculate_movement_combinations(self, chord: str) -> float:
    """Calculate score for movement combinations in chord."""
    if not hasattr(self, "keyboard_layout"):
        self.keyboard_layout = self._load_keyboard_layout(
            Path(self.config.keylayout_csv_file)
        )

    score = 0.0
    chord_letters = [
        self.keyboard_layout[c] for c in chord if c in self.keyboard_layout
    ]
    used_fingers = [l.finger for l in chord_letters]

    # Check for same finger usage
    for i in range(1, len(used_fingers)):
        if used_fingers[i] == used_fingers[i - 1]:
            score += self.weights["same_finger_double"]
            if i > 1 and used_fingers[i] == used_fingers[i - 2]:
                score += self.weights["same_finger_triple"]

    # Check for awkward combinations
    for i in range(1, len(used_fingers)):
        curr, prev = used_fingers[i], used_fingers[i - 1]

        # Pinky-ring stretches
        if (
            prev in (FingerIndex.LPINKY, FingerIndex.RPINKY)
            and curr in (FingerIndex.LRING, FingerIndex.RRING)
        ) or (
            prev in (FingerIndex.LRING, FingerIndex.RRING)
            and curr in (FingerIndex.LPINKY, FingerIndex.RPINKY)
        ):
            score += self.weights["pinky_ring_stretch"]

        # Ring-middle scissors
        if (
            prev in (FingerIndex.LRING, FingerIndex.RRING)
            and curr in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
        ) or (
            prev in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
            and curr in (FingerIndex.LRING, FingerIndex.RRING)
        ):
            score += self.weights["ring_middle_scissor"]

        # Middle-index stretches
        if (
            prev in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
            and curr in (FingerIndex.LINDEX, FingerIndex.RINDEX)
        ) or (
            prev in (FingerIndex.LINDEX, FingerIndex.RINDEX)
            and curr in (FingerIndex.LMIDDLE, FingerIndex.RMIDDLE)
        ):
            score += self.weights["middle_index_stretch"]

    return score


def calculate_visual_similarity(self, word: str, chord: str) -> float:
    """Calculate visual similarity between word and chord."""
    # Implementation would use string similarity metrics
    return 0.0  # Placeholder


def calculate_phonetic_similarity(self, word: str, chord: str) -> float:
    """Calculate phonetic similarity between word and chord."""
    # Implementation would use phonetic algorithms
    return 0.0  # Placeholder


def calculate_missing_letters_score(self, word: str, chord: str) -> float:
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


def calculate_extra_letters_score(self, word: str, chord: str) -> float:
    """Calculate score for extra letters in chord not in word."""
    return (
        len(set(chord.lower()) - set(word.lower())) * self.weights["additional_letter"]
    )


def calculate_ngram_preservation(self, word: str, chord: str) -> float:
    """Calculate how well the chord preserves word n-grams."""
    # Implementation would analyze bigrams and trigrams
    return 0.0  # Placeholder


def calculate_word_priority(self, word: str) -> float:
    """Calculate priority score for word."""
    # Implementation would consider word frequency and difficulty
    return 0.0  # Placeholder


def calculate_finger_utilization(self, assignments: Dict[str, str]) -> Dict[str, float]:
    """Calculate finger utilization statistics."""
    # Implementation would track finger usage
    return {}  # Placeholder


def calculate_hand_balance(self, assignments: Dict[str, str]) -> float:
    """Calculate hand balance score."""
    # Implementation would analyze left/right hand usage
    return 0.0  # Placeholder


def calculate_chord_pattern_consistency(self, assignments: Dict[str, str]) -> float:
    """Calculate consistency of chord patterns."""
    # Implementation would analyze pattern similarities
    return 0.0  # Placeholder


def calculate_sequence_difficulty(self, assignments: Dict[str, str]) -> float:
    """Calculate difficulty of chord sequences."""
    # Implementation would analyze transitions between chords
    return 0.0  # Placeholder


def calculate_summary_metrics(
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
