from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional


class FingerIndex(Enum):
    """Enum for finger positions on keyboard"""

    LPINKY = auto()
    LRING = auto()
    LMIDDLE = auto()
    LINDEX = auto()
    LTHUMB = auto()
    RTHUMB = auto()
    RINDEX = auto()
    RMIDDLE = auto()
    RRING = auto()
    RPINKY = auto()


@dataclass
class LetterData:
    """Store relevant information about a letter"""

    finger: FingerIndex
    vertical_distance_to_home_row: int
    horizontal_distance_to_home_row: int
    finger_to_left: Optional[FingerIndex]
    finger_to_right: Optional[FingerIndex]


@dataclass
class WordData:
    """Store preprocessed word data"""

    original: str
    lower: str
    length: int


def _load_keyboard_layout(self, layout_file: Path) -> Dict[str, LetterData]:
    """Load keyboard layout from CSV file.

    Args:
        layout_file: Path to the CSV layout file

    Returns:
        Dictionary mapping letters to their keyboard position data
    """
    # TODO: separate into something like a definitions module
    mapping = {}
    finger_mapping = {
        "lp": FingerIndex.LPINKY,
        "lr": FingerIndex.LRING,
        "lm": FingerIndex.LMIDDLE,
        "li": FingerIndex.LINDEX,
        "ri": FingerIndex.RINDEX,
        "rm": FingerIndex.RMIDDLE,
        "rr": FingerIndex.RRING,
        "rp": FingerIndex.RPINKY,
    }

    with open(layout_file, encoding="utf-8") as file:
        rows = [line.strip().split(",") for line in file]
        layout_rows = rows[:3]
        finger_map_rows = rows[3:6]
        vertical_map_rows = rows[6:9]
        horizontal_map_rows = rows[9:12]

        # TODO: add support for other layouts from algorithm1
        no_left_fingers = {FingerIndex.LPINKY, FingerIndex.RINDEX}
        no_right_fingers = {FingerIndex.RPINKY, FingerIndex.LINDEX}

        for row_idx, row in enumerate(layout_rows):
            for col_idx, letter in enumerate(row):
                if letter == "-":
                    continue
                current_finger = finger_mapping[finger_map_rows[row_idx][col_idx]]
                mapping[letter] = LetterData(
                    current_finger,
                    int(vertical_map_rows[row_idx][col_idx]),
                    int(horizontal_map_rows[row_idx][col_idx]),
                    (
                        None
                        if current_finger in no_left_fingers
                        else finger_mapping[finger_map_rows[row_idx][col_idx - 1]]
                    ),
                    (
                        None
                        if current_finger in no_right_fingers
                        else finger_mapping[finger_map_rows[row_idx][col_idx + 1]]
                    ),
                )
    return mapping
