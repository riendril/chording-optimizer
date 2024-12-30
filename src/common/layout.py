from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional


class KeyboardFormat(Enum):
    """Enum for the currently supported keyboard types"""

    MATRIX = auto()
    MASTER_FORGE_3D = auto()


class Finger(Enum):
    """Enum for finger positions on keyboard"""

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


@dataclass
class LetterData:
    """Store relevant information about a letter"""

    finger: Finger
    vertical_distance_to_home_row: int
    horizontal_distance_to_home_row: int
    finger_to_left: Optional[Finger]
    finger_to_right: Optional[Finger]


def load_keyboard_layout(layout_file: Path) -> Dict[str, LetterData]:
    """Load keyboard layout from CSV file.

    Args:
        layout_file: Path to the CSV layout file

    Returns:
        Dictionary mapping letters to their keyboard position data
    """
    # TODO: put into something like an enum
    mapping = {}
    finger_mapping = {
        "lp": Finger.L_PINKY,
        "lr": Finger.L_RING,
        "lm": Finger.L_MIDDLE,
        "li": Finger.L_INDEX,
        "ri": Finger.R_INDEX,
        "rm": Finger.R_MIDDLE,
        "rr": Finger.R_RING,
        "rp": Finger.R_PINKY,
    }

    with open(layout_file, encoding="utf-8") as file:
        rows = [line.strip().split(",") for line in file]
        layout_rows = rows[:3]
        finger_map_rows = rows[3:6]
        vertical_map_rows = rows[6:9]
        horizontal_map_rows = rows[9:12]

        # TODO: Check if should be configured
        has_no_left_fingers = {
            Finger.L_PINKY,
            Finger.R_INDEX,
            Finger.L_THUMB,
            Finger.R_THUMB,
        }
        has_no_right_fingers = {
            Finger.R_PINKY,
            Finger.L_INDEX,
            Finger.L_THUMB,
            Finger.R_THUMB,
        }

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
                        if current_finger in has_no_left_fingers
                        else finger_mapping[finger_map_rows[row_idx][col_idx - 1]]
                    ),
                    (
                        None
                        if current_finger in has_no_right_fingers
                        else finger_mapping[finger_map_rows[row_idx][col_idx + 1]]
                    ),
                )
    return mapping
