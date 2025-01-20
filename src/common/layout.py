import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


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


class KeyPosition(NamedTuple):
    """Position and finger assignment of a key on the input device"""

    finger: Finger
    vertical_distance_to_resting_position: int
    horizontal_distance_to_resting_position: int
    finger_to_left: Optional[Finger]
    finger_to_right: Optional[Finger]


def parse_key_entry(line: str) -> Optional[Tuple[str, List[int]]]:
    """Parse a key entry line from the layout file.
    Returns None for non-key lines (comments, empty lines, etc)"""
    # Remove any trailing comments
    line = line.split("#")[0].strip()

    # Skip empty lines, section markers, and closing braces
    if not line or line.endswith("{") or line == "}":
        return None

    # Try to match key entry pattern
    match = re.match(r'\s*"([^"]+)"\s*\[([^]]+)\]', line)
    if not match:
        return None

    char = match.group(1)
    try:
        positions = [int(x.strip()) for x in match.group(2).split(",")]
        if len(positions) == 2:
            return char, positions
    except ValueError:
        pass

    return None


def get_finger_enum(hand: str, finger: str) -> Finger:
    """Convert hand and finger string to Finger enum"""
    prefix = "L_" if hand == "LEFT_HAND" else "R_"
    return Finger[prefix + finger.upper()]


def load_input_device_layout(layout_file: Path) -> Dict[str, KeyPosition]:
    """Load key positions and finger assignments from layout file."""
    mapping: Dict[str, KeyPosition] = {}
    current_hand = None
    current_finger = None

    # Track which fingers actually have keys assigned
    finger_positions: Dict[Finger, List[Tuple[str, List[int]]]] = {}
    active_fingers: Dict[str, Set[Finger]] = {"LEFT_HAND": set(), "RIGHT_HAND": set()}

    with open(layout_file, encoding="utf-8") as file:
        for line in file:
            # Skip empty lines and pure comments
            if not line.strip() or line.strip().startswith("#"):
                continue

            # Check for hand/finger sections
            if line.strip().endswith("{"):
                section = (
                    line.split("#")[0].strip()[:-1].strip()
                )  # Remove comment and brace
                if section in ["LEFT_HAND", "RIGHT_HAND"]:
                    current_hand = section
                    current_finger = None
                elif current_hand and section in [
                    "PINKY",
                    "RING",
                    "MIDDLE",
                    "INDEX",
                    "THUMB",
                ]:
                    current_finger = section
                continue

            if line.strip() == "}":
                continue

            # Parse key entries
            if current_hand and current_finger:
                key_entry = parse_key_entry(line)
                if key_entry:
                    char, positions = key_entry
                    finger = get_finger_enum(current_hand, current_finger)

                    if finger not in finger_positions:
                        finger_positions[finger] = []
                    finger_positions[finger].append((char, positions))
                    active_fingers[current_hand].add(finger)

    # For each hand, get ordered list of active fingers
    left_active = sorted([f for f in active_fingers["LEFT_HAND"]], key=lambda x: x.name)
    right_active = sorted(
        [f for f in active_fingers["RIGHT_HAND"]], key=lambda x: x.name
    )

    # Create the mappings with correct neighbor assignments
    for finger, entries in finger_positions.items():
        is_left_hand = finger.name.startswith("L_")
        active_list = left_active if is_left_hand else right_active

        if finger in active_list:
            idx = active_list.index(finger)
            finger_to_left = active_list[idx - 1] if idx > 0 else None
            finger_to_right = (
                active_list[idx + 1] if idx < len(active_list) - 1 else None
            )

            for char, positions in entries:
                mapping[char] = KeyPosition(
                    finger=finger,
                    vertical_distance_to_resting_position=positions[0],
                    horizontal_distance_to_resting_position=positions[1],
                    finger_to_left=finger_to_left,
                    finger_to_right=finger_to_right,
                )

    return mapping
