"""
Standalone metrics for individual chord evaluation.
Usage:
    calculator = StandaloneMetricCalculator(config)
    chord_data = ChordData(...)
    standalone_cost = calculator.calculate(chord_data)
"""

from math import sqrt
from typing import Callable, Dict, List, Optional

from src.common.benchmarking import Benchmark
from src.common.config import GeneratorConfig
from src.common.layout import Finger
from src.common.shared_types import ChordData, StandaloneMetrics, StandaloneMetricType

# Type alias for standalone metric calculation functions
StandaloneMetricFn = Callable[[ChordData], float]


class StandaloneMetricCalculator:
    """Calculates metrics for individual chords using configured weights"""

    def __init__(self, config: GeneratorConfig, benchmark: Optional[Benchmark] = None):
        """Initialize calculator with configuration weights"""
        self.weights = config.standalone_weights.weights
        self.benchmark = benchmark
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
        """Calculate all metrics for a single chord"""
        costs = {}
        for metric_type in StandaloneMetricType:
            if self.benchmark is not None:
                self.benchmark.start_metric_calculation(metric_type.name)
                raw_cost = self.metric_functions[metric_type](chord)
                self.benchmark.end_metric_calculation()
            else:
                raw_cost = self.metric_functions[metric_type](chord)
            costs[metric_type] = raw_cost * self.weights[metric_type]

        return StandaloneMetrics(costs=costs)

    def _calc_chord_length(self, chord: ChordData) -> float:
        """Calculate length cost"""
        return float(chord.length)

    def _calc_horizontal_stretch(self, chord: ChordData) -> float:
        """Calculate horizontal stretching cost as sum of horizontal stretches"""
        stretch_sum = 1.0
        for key in chord.keys:
            stretch_sum *= 1 + max(0, key.horizontal_distance_to_resting_position)
        return stretch_sum

    def _calc_horizontal_pinch(self, chord: ChordData) -> float:
        """Calculate horizontal pinching cost as sum of horizontal pinches"""
        pinch_sum = 0.0
        for key in chord.keys:
            pinch_sum += min(0, key.horizontal_distance_to_resting_position)
        return pinch_sum

    def _calc_vertical_stretch(self, chord: ChordData) -> float:
        """Calculate vertical stretching cost as sum of vertical stretches"""
        stretch_sum = 0.0
        for key in chord.keys:
            stretch_sum += max(0, key.vertical_distance_to_resting_position)
        return stretch_sum

    def _calc_vertical_pinch(self, chord: ChordData) -> float:
        """Calculate vertical pinching cost as sum of vertical pinches"""
        pinch_sum = 0.0
        for key in chord.keys:
            pinch_sum += min(0, key.vertical_distance_to_resting_position)
        return pinch_sum

    def _calc_diagonal_stretch(self, chord: ChordData) -> float:
        """Calculate diagonal stretching cost as sum of diagonal stretches"""
        stretch_sum = 0.0
        for key in chord.keys:
            if key.vertical_distance_to_resting_position > 0:
                if key.horizontal_distance_to_resting_position > 0:
                    stretch_sum += sqrt(
                        key.vertical_distance_to_resting_position**2
                        + key.horizontal_distance_to_resting_position**2
                    )
        return stretch_sum

    def _calc_diagonal_pinch(self, chord: ChordData) -> float:
        """Calculate diagonal pinching cost"""
        max_pinch = 0
        for i, key1 in enumerate(chord.keys):
            for key2 in chord.keys[i + 1 :]:
                if (
                    key1.finger_to_right == key2.finger
                    or key2.finger_to_right == key1.finger
                ):
                    h_pinch = abs(
                        key1.horizontal_distance_to_resting_position
                        - key2.horizontal_distance_to_resting_position
                    )
                    v_pinch = abs(
                        key1.vertical_distance_to_resting_position
                        - key2.vertical_distance_to_resting_position
                    )
                    diagonal = (h_pinch**2 + v_pinch**2) ** 0.5
                    max_pinch = max(max_pinch, diagonal)
        return float(max_pinch)

    def _calc_same_finger_double_adjacent(self, chord: ChordData) -> float:
        """Calculate cost for adjacent keys pressed by same finger"""
        count = 0
        for i, key1 in enumerate(chord.keys):
            for key2 in chord.keys[i + 1 :]:
                if (
                    key1.finger == key2.finger
                    and abs(
                        key1.vertical_distance_to_resting_position
                        - key2.vertical_distance_to_resting_position
                    )
                    == 1
                ):
                    count += 1
        return float(count)

    def _calc_same_finger_double_gap(self, chord: ChordData) -> float:
        """Calculate cost for non-adjacent keys pressed by same finger"""
        count = 0
        for i, key1 in enumerate(chord.keys):
            for key2 in chord.keys[i + 1 :]:
                if (
                    key1.finger == key2.finger
                    and abs(
                        key1.vertical_distance_to_resting_position
                        - key2.vertical_distance_to_resting_position
                    )
                    > 1
                ):
                    count += 1
        return float(count)

    def _calc_same_finger_triple(self, chord: ChordData) -> float:
        """Calculate cost for three keys pressed by same finger"""
        finger_counts = {}
        for key in chord.keys:
            finger_counts[key.finger] = finger_counts.get(key.finger, 0) + 1
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
        for key in chord.keys:
            if key.finger not in finger_positions:
                finger_positions[key.finger] = []
            finger_positions[key.finger].append(
                (
                    key.vertical_distance_to_resting_position,
                    key.horizontal_distance_to_resting_position,
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
        for i, key1 in enumerate(chord.keys):
            for key2 in chord.keys[i + 1 :]:
                if (
                    key1.finger_to_right == key2.finger
                    or key2.finger_to_right == key1.finger
                ):
                    # Check for partial crossing movement
                    if (
                        key1.vertical_distance_to_resting_position > 0
                        and key2.vertical_distance_to_resting_position < 0
                    ) or (
                        key1.vertical_distance_to_resting_position < 0
                        and key2.vertical_distance_to_resting_position > 0
                    ):
                        count += 1
        return float(count)

    def _calc_horizontal_stretch_double(self, chord: ChordData) -> float:
        """Calculate cost for horizontal stretching between two fingers"""
        max_stretch = 0
        for i, key1 in enumerate(chord.keys):
            for key2 in chord.keys[i + 1 :]:
                if key1.finger.name[0] == key2.finger.name[0]:  # Same hand
                    stretch = abs(
                        key1.horizontal_distance_to_resting_position
                        - key2.horizontal_distance_to_resting_position
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

    def _is_crossing_movement(self, positions: List[tuple]) -> bool:
        """Check if a sequence of finger positions creates a crossing movement.

        Args:
            positions: List of (vertical, horizontal) positions for each finger

        Returns:
            True if the movements cross (create scissors), False otherwise
        """
        # Check if any adjacent pair of fingers creates a crossing pattern
        for i in range(len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]

            # Check if vertical movements cross over
            if (pos1[0] > 0 and pos2[0] < 0) or (pos1[0] < 0 and pos2[0] > 0):
                # Also verify horizontal positions create a potential crossing
                if abs(pos1[1] - pos2[1]) > 0:
                    return True
        return False

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
        for i, key1 in enumerate(chord.keys):
            if key1.finger not in target_fingers:
                continue
            for key2 in chord.keys[i + 1 :]:
                if key2.finger not in target_fingers:
                    continue
                if key1.finger.name[0] == key2.finger.name[0]:  # Same hand
                    # Check for crossing movement
                    if (
                        key1.vertical_distance_to_resting_position > 0
                        and key2.vertical_distance_to_resting_position < 0
                    ) or (
                        key1.vertical_distance_to_resting_position < 0
                        and key2.vertical_distance_to_resting_position > 0
                    ):
                        count += 1
        return float(count)
