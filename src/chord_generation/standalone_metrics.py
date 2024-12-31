"""
Standalone metrics for individual chord evaluation.
Usage:
    calculator = StandaloneMetricCalculator(config)
    chord_data = ChordData(...)
    metrics = calculator.calculate(chord_data)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple

from src.common.config import GeneratorConfig
from src.common.layout import Finger, LetterData
from src.common.shared_types import ChordData, StandaloneMetrics, StandaloneMetricType

# Type alias for standalone metric calculation functions
StandaloneMetricFn = Callable[[ChordData], float]


class StandaloneMetricCalculator:
    """Calculates metrics for individual chords using configured weights"""

    def __init__(self, config: GeneratorConfig):
        """Initialize calculator with configuration weights"""
        self.weights = config.standalone_weights.weights
        # Map metric types to their calculation functions
        self.metric_functions: Dict[StandaloneMetricType, StandaloneMetricFn] = {
            StandaloneMetricType.CHORD_LENGTH: self._calc_chord_length,
            StandaloneMetricType.HORIZONTAL_STRETCH: self._calc_horizontal_stretch,
            StandaloneMetricType.VERTICAL_STRETCH: self._calc_vertical_stretch,
            StandaloneMetricType.DIAGONAL_STRETCH: self._calc_diagonal_stretch,
            StandaloneMetricType.HORIZONTAL_PINCH: self._calc_horizontal_pinch,
            StandaloneMetricType.VERTICAL_PINCH: self._calc_vertical_pinch,
            StandaloneMetricType.DIAGONAL_PINCH: self._calc_diagonal_pinch,
            StandaloneMetricType.SAME_FINGER_DOUBLE_ADJACENT: self._calc_same_finger_double_adjacent,
            StandaloneMetricType.SAME_FINGER_DOUBLE_GAP: self._calc_same_finger_double_gap,
            StandaloneMetricType.SAME_FINGER_TRIPLE: self._calc_same_finger_triple,
            StandaloneMetricType.FULL_SCISSOR_DOUBLE: self._calc_full_scissor_double,
            StandaloneMetricType.FULL_SCISSOR_TRIPLE: self._calc_full_scissor_triple,
            StandaloneMetricType.FULL_SCISSOR_QUADRUPLE: self._calc_full_scissor_quadruple,
            StandaloneMetricType.FULL_SCISSOR_QUINTUPLE: self._calc_full_scissor_quintuple,
            StandaloneMetricType.HALF_SCISSOR_DOUBLE: self._calc_half_scissor_double,
            StandaloneMetricType.HORIZONTAL_STRETCH_DOUBLE: self._calc_horizontal_stretch_double,
            StandaloneMetricType.PINKY_RING_SCISSOR: self._calc_pinky_ring_scissor,
            StandaloneMetricType.RING_INDEX_SCISSOR: self._calc_ring_index_scissor,
        }

    def calculate(self, chord: ChordData) -> StandaloneMetrics:
        """Calculate all metrics for a single chord

        Args:
            chord: Preprocessed chord data

        Returns:
            Complete set of weighted metric costs for the chord
        """
        costs = {}
        for metric_type in StandaloneMetricType:
            raw_cost = self.metric_functions[metric_type](chord)
            costs[metric_type] = raw_cost * self.weights[metric_type]

        return StandaloneMetrics(costs=costs)

    def _calc_chord_length(self, chord: ChordData) -> float:
        """Calculate base cost based on chord length"""
        return float(chord.length)

    def _calc_horizontal_stretch(self, chord: ChordData) -> float:
        """Calculate horizontal stretching cost"""
        max_stretch = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if letter1.finger.name[0] == letter2.finger.name[0]:  # Same hand
                    stretch = abs(
                        letter1.horizontal_distance_to_home_row
                        - letter2.horizontal_distance_to_home_row
                    )
                    max_stretch = max(max_stretch, stretch)
        return float(max_stretch)

    def _calc_vertical_stretch(self, chord: ChordData) -> float:
        """Calculate vertical stretching cost"""
        max_stretch = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if letter1.finger.name[0] == letter2.finger.name[0]:  # Same hand
                    stretch = abs(
                        letter1.vertical_distance_to_home_row
                        - letter2.vertical_distance_to_home_row
                    )
                    max_stretch = max(max_stretch, stretch)
        return float(max_stretch)

    def _calc_diagonal_stretch(self, chord: ChordData) -> float:
        """Calculate diagonal stretching cost"""
        max_stretch = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if letter1.finger.name[0] == letter2.finger.name[0]:  # Same hand
                    h_stretch = abs(
                        letter1.horizontal_distance_to_home_row
                        - letter2.horizontal_distance_to_home_row
                    )
                    v_stretch = abs(
                        letter1.vertical_distance_to_home_row
                        - letter2.vertical_distance_to_home_row
                    )
                    diagonal = (h_stretch**2 + v_stretch**2) ** 0.5
                    max_stretch = max(max_stretch, diagonal)
        return float(max_stretch)

    def _calc_horizontal_pinch(self, chord: ChordData) -> float:
        """Calculate horizontal pinching cost"""
        max_pinch = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if (
                    letter1.finger_to_right == letter2.finger
                    or letter2.finger_to_right == letter1.finger
                ):
                    pinch = abs(
                        letter1.horizontal_distance_to_home_row
                        - letter2.horizontal_distance_to_home_row
                    )
                    max_pinch = max(max_pinch, pinch)
        return float(max_pinch)

    def _calc_vertical_pinch(self, chord: ChordData) -> float:
        """Calculate vertical pinching cost"""
        max_pinch = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if (
                    letter1.finger_to_right == letter2.finger
                    or letter2.finger_to_right == letter1.finger
                ):
                    pinch = abs(
                        letter1.vertical_distance_to_home_row
                        - letter2.vertical_distance_to_home_row
                    )
                    max_pinch = max(max_pinch, pinch)
        return float(max_pinch)

    def _calc_diagonal_pinch(self, chord: ChordData) -> float:
        """Calculate diagonal pinching cost"""
        max_pinch = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if (
                    letter1.finger_to_right == letter2.finger
                    or letter2.finger_to_right == letter1.finger
                ):
                    h_pinch = abs(
                        letter1.horizontal_distance_to_home_row
                        - letter2.horizontal_distance_to_home_row
                    )
                    v_pinch = abs(
                        letter1.vertical_distance_to_home_row
                        - letter2.vertical_distance_to_home_row
                    )
                    diagonal = (h_pinch**2 + v_pinch**2) ** 0.5
                    max_pinch = max(max_pinch, diagonal)
        return float(max_pinch)

    def _calc_same_finger_double_adjacent(self, chord: ChordData) -> float:
        """Calculate cost for adjacent keys pressed by same finger"""
        count = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if (
                    letter1.finger == letter2.finger
                    and abs(
                        letter1.vertical_distance_to_home_row
                        - letter2.vertical_distance_to_home_row
                    )
                    == 1
                ):
                    count += 1
        return float(count)

    def _calc_same_finger_double_gap(self, chord: ChordData) -> float:
        """Calculate cost for non-adjacent keys pressed by same finger"""
        count = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if (
                    letter1.finger == letter2.finger
                    and abs(
                        letter1.vertical_distance_to_home_row
                        - letter2.vertical_distance_to_home_row
                    )
                    > 1
                ):
                    count += 1
        return float(count)

    def _calc_same_finger_triple(self, chord: ChordData) -> float:
        """Calculate cost for three keys pressed by same finger"""
        finger_counts = {}
        for letter in chord.letters:
            finger_counts[letter.finger] = finger_counts.get(letter.finger, 0) + 1
        return float(sum(1 for count in finger_counts.values() if count >= 3))

    def _calc_full_scissor_double(self, chord: ChordData) -> float:
        """Calculate cost for full scissor movement between two fingers"""
        return self._calc_scissor_movement(chord, 2)

    def _calc_full_scissor_triple(self, chord: ChordData) -> float:
        """Calculate cost for full scissor movement between three fingers"""
        return self._calc_scissor_movement(chord, 3)

    def _calc_full_scissor_quadruple(self, chord: ChordData) -> float:
        """Calculate cost for full scissor movement between four fingers"""
        return self._calc_scissor_movement(chord, 4)

    def _calc_full_scissor_quintuple(self, chord: ChordData) -> float:
        """Calculate cost for full scissor movement between five fingers"""
        return self._calc_scissor_movement(chord, 5)

    def _calc_scissor_movement(self, chord: ChordData, n_fingers: int) -> float:
        """Helper method to calculate scissor movements

        Args:
            chord: Preprocessed chord data
            n_fingers: Number of fingers to check for scissor movement

        Returns:
            Count of scissor movements found
        """
        finger_positions = {}
        for letter in chord.letters:
            if letter.finger not in finger_positions:
                finger_positions[letter.finger] = []
            finger_positions[letter.finger].append(
                (
                    letter.vertical_distance_to_home_row,
                    letter.horizontal_distance_to_home_row,
                )
            )

        # Count sequences of n adjacent fingers with crossing movements
        count = 0
        fingers = list(Finger)
        for i in range(len(fingers) - n_fingers + 1):
            sequence = fingers[i : i + n_fingers]
            if all(f in finger_positions for f in sequence):
                # Check if movements cross
                positions = [finger_positions[f][0] for f in sequence]
                if self._is_crossing_movement(positions):
                    count += 1
        return float(count)

    def _calc_half_scissor_double(self, chord: ChordData) -> float:
        """Calculate cost for half scissor movement between two fingers"""
        count = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if (
                    letter1.finger_to_right == letter2.finger
                    or letter2.finger_to_right == letter1.finger
                ):
                    # Check for partial crossing movement
                    if (
                        letter1.vertical_distance_to_home_row > 0
                        and letter2.vertical_distance_to_home_row < 0
                    ) or (
                        letter1.vertical_distance_to_home_row < 0
                        and letter2.vertical_distance_to_home_row > 0
                    ):
                        count += 1
        return float(count)

    def _calc_horizontal_stretch_double(self, chord: ChordData) -> float:
        """Calculate cost for horizontal stretching between two fingers"""
        max_stretch = 0
        for i, letter1 in enumerate(chord.letters):
            for letter2 in chord.letters[i + 1 :]:
                if letter1.finger.name[0] == letter2.finger.name[0]:  # Same hand
                    stretch = abs(
                        letter1.horizontal_distance_to_home_row
                        - letter2.horizontal_distance_to_home_row
                    )
                    if stretch >= 2:  # Only count significant stretches
                        max_stretch = max(max_stretch, stretch)
        return float(max_stretch)

    def _calc_pinky_ring_scissor(self, chord: ChordData) -> float:
        """Calculate cost for scissor movement between pinky and ring fingers"""
        return self._calc_specific_finger_scissor(
            chord, [Finger.L_PINKY, Finger.L_RING, Finger.R_PINKY, Finger.R_RING]
        )

    def _calc_ring_index_scissor(self, chord: ChordData) -> float:
        """Calculate cost for scissor movement between ring and index fingers"""
        return self._calc_specific_finger_scissor(
            chord, [Finger.L_RING, Finger.L_INDEX, Finger.R_RING, Finger.R_INDEX]
        )

    def _calc_specific_finger_scissor(
        self, chord: ChordData, target_fingers: List[Finger]
    ) -> float:
        """Helper method to calculate scissor movements for specific fingers

        Args:
            chord: Preprocessed chord data
            target_fingers: List of fingers to check for scissor movements

        Returns:
            Count of scissor movements between specified fingers
        """
        count = 0
        for i, letter1 in enumerate(chord.letters):
            if letter1.finger not in target_fingers:
                continue
            for letter2 in chord.letters[i + 1 :]:
                if letter2.finger not in target_fingers:
                    continue
                if letter1.finger.name[0] == letter2.finger.name[0]:  # Same hand
                    # Check for crossing movement
                    if (
                        letter1.vertical_distance_to_home_row > 0
                        and letter2.vertical_distance_to_home_row < 0
                    ) or (
                        letter1.vertical_distance_to_home_row < 0
                        and letter2.vertical_distance_to_home_row > 0
                    ):
                        count += 1
        return float(count)
