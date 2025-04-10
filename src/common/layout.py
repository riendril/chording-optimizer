"""Layout loader for keyboard configurations

This module handles loading and processing of YAML-based keyboard layouts.
It supports layouts with explicit comfort matrices or can generate them
based on finger position offsets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.common.shared_types import Finger, KeyPosition

logger = logging.getLogger(__name__)


def load_layout_from_yaml(layout_file: Path) -> Dict:
    """Load keyboard layout from YAML file"""
    with open(layout_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_layout_to_yaml(layout_data: Dict, layout_file: Path) -> None:
    """Save keyboard layout to YAML file"""
    with open(layout_file, "w", encoding="utf-8") as f:
        yaml.dump(layout_data, f, default_flow_style=False, sort_keys=False)


def create_key_positions(layout_data: Dict) -> Dict[str, KeyPosition]:
    """Create KeyPosition objects from layout data"""
    # Extract position and finger data
    positions = layout_data.get("positions", {})
    fingers = layout_data.get("fingers", {})

    # Create key positions
    key_positions = {}
    for key, pos in positions.items():
        row, col = pos
        finger_name = fingers.get(key)

        if not finger_name:
            continue  # Skip keys without finger assignments

        # Get finger enum
        finger = Finger[finger_name]

        # Find adjacent finger assignments
        finger_to_left = None
        finger_to_right = None

        # Check adjacent keys in the same row to infer finger relationships
        # Find finger to left (rough approximation)
        for other_key, other_pos in positions.items():
            other_row, other_col = other_pos
            other_finger_name = fingers.get(other_key)

            if other_finger_name and other_row == row:
                if other_col == col - 1:  # Key to the left
                    finger_to_left = Finger[other_finger_name]
                elif other_col == col + 1:  # Key to the right
                    finger_to_right = Finger[other_finger_name]

        # Create KeyPosition object
        key_positions[key] = KeyPosition(
            finger=finger,
            vertical_distance_to_resting_position=row,
            horizontal_distance_to_resting_position=col,
            finger_to_left=finger_to_left,
            finger_to_right=finger_to_right,
        )

    return key_positions


def generate_comfort_matrix(
    positions: Dict[str, List[float]], fingers: Dict[str, str]
) -> Dict[str, float]:
    """Generate comfort values based on finger positions

    Args:
        positions: Dict of key to [row, col] positions
        fingers: Dict of key to finger name

    Returns:
        Dict of key to comfort value (lower is better)
    """
    comfort = {}

    # Find home row positions for each finger
    finger_home_positions = {}
    for key, finger_name in fingers.items():
        row, col = positions[key]
        if finger_name not in finger_home_positions and row == 0:
            # Assuming row 0 is the home row
            finger_home_positions[finger_name] = (row, col)

    # Assign comfort values based on distance from home position and finger
    for key, pos in positions.items():
        row, col = pos
        finger = fingers.get(key)
        if not finger:
            comfort[key] = 10  # Default high value for unassigned keys
            continue

        # Base comfort depends on row distance from home row
        row_distance = abs(row)
        base_comfort = row_distance * 2

        # Add penalty for non-index/middle fingers
        if finger in ("L_PINKY", "R_PINKY"):
            base_comfort += 3
        elif finger in ("L_RING", "R_RING"):
            base_comfort += 1

        # Add penalty for extreme horizontal positions
        if col < -3 or col > 3:
            base_comfort += 2
        elif col < -2 or col > 2:
            base_comfort += 1

        # Add thumb penalty if used for non-space key
        if finger in ("L_THUMB", "R_THUMB") and key != " ":
            base_comfort += 1

        # Ensure minimum comfort value is 0
        comfort[key] = max(0, base_comfort)

    return comfort


def convert_old_layout_to_yaml(layout_file: Path) -> Dict:
    """Convert layout files using the old format to the new YAML structure

    Args:
        layout_file: Path to layout file in old format

    Returns:
        Dict with the converted layout data
    """
    with open(layout_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract metadata from comments
    metadata = {}
    for line in content.split("\n"):
        if line.startswith("# "):
            parts = line[2:].split(":", 1)
            if len(parts) == 2:
                key, value = parts
                metadata[key.strip()] = value.strip()

    # Initialize structure
    layout_data = {
        "name": metadata.get("KeyLayout", layout_file.stem),
        "type": metadata.get("Keyboard Format", "unknown"),
        "description": f"Converted from {layout_file.name}",
        "positions": {},
        "fingers": {},
    }

    # Parse the content for key positions and finger assignments
    # This is a rough parser for the specific format provided
    current_hand = None
    current_finger = None

    for line in content.split("\n"):
        line = line.strip()

        # Detect hand section
        if "LEFT_HAND" in line:
            current_hand = "L"
        elif "RIGHT_HAND" in line:
            current_hand = "R"

        # Detect finger section
        if "PINKY" in line and "{" in line:
            current_finger = "PINKY"
        elif "RING" in line and "{" in line:
            current_finger = "RING"
        elif "MIDDLE" in line and "{" in line:
            current_finger = "MIDDLE"
        elif "INDEX" in line and "{" in line:
            current_finger = "INDEX"
        elif "THUMB" in line and "{" in line:
            current_finger = "THUMB"

        # Extract key assignments with positions
        if current_hand and current_finger and '"' in line and "[" in line:
            # Extract the key and position
            key_part = line.split('"')[1]
            pos_part = line.split("[")[1].split("]")[0]

            key = key_part
            # Parse position offsets
            try:
                vertical, horizontal = map(float, pos_part.split(","))

                # Store position as row, col where row 0 is home row
                # For this format, we're keeping the home row relative positions
                layout_data["positions"][key] = [vertical, horizontal]

                # Store finger assignment
                layout_data["fingers"][key] = f"{current_hand}_{current_finger}"
            except ValueError:
                logger.warning(
                    f"Could not parse position for key {key} in {layout_file}"
                )

    # Generate comfort matrix
    layout_data["comfort"] = generate_comfort_matrix(
        layout_data["positions"], layout_data["fingers"]
    )

    return layout_data


def load_keyboard_layout(
    layout_file: Path, force_regenerate_comfort: bool = False
) -> Dict:
    """Load keyboard layout with all data needed for analysis

    Args:
        layout_file: Path to the layout file
        force_regenerate_comfort: If True, regenerate comfort matrix even if it exists

    Returns:
        Dictionary with complete layout data
    """
    # Check if it's a legacy layout file
    if layout_file.suffix == ".layout":
        logger.info(f"Converting legacy layout file: {layout_file}")
        layout_data = convert_old_layout_to_yaml(layout_file)

        # Save converted layout as YAML
        new_file = layout_file.with_suffix(".yaml")
        save_layout_to_yaml(layout_data, new_file)
        logger.info(f"Saved converted layout to {new_file}")

    else:
        # Load existing YAML file
        layout_data = load_layout_from_yaml(layout_file)

    # Generate comfort matrix if not present or regeneration forced
    if "comfort" not in layout_data or force_regenerate_comfort:
        layout_data["comfort"] = generate_comfort_matrix(
            layout_data["positions"], layout_data["fingers"]
        )
        # Save updated layout with comfort matrix
        save_layout_to_yaml(layout_data, layout_file)
        logger.info(f"Generated and saved comfort matrix for {layout_file}")

    # Create KeyPosition objects
    key_positions = create_key_positions(layout_data)

    # Return complete layout data
    return {
        "name": layout_data.get("name", layout_file.stem),
        "type": layout_data.get("type", "unknown"),
        "description": layout_data.get("description", ""),
        "positions": layout_data.get("positions", {}),
        "comfort": layout_data.get("comfort", {}),
        "fingers": layout_data.get("fingers", {}),
        "key_positions": key_positions,
    }


def convert_all_layouts(layouts_dir: Path) -> None:
    """Convert all legacy layout files in a directory to YAML format

    Args:
        layouts_dir: Directory containing layout files
    """
    layout_files = list(layouts_dir.glob("*.layout"))
    logger.info(f"Found {len(layout_files)} legacy layout files to convert")

    for layout_file in layout_files:
        try:
            load_keyboard_layout(layout_file)  # This will convert and save
        except Exception as e:
            logger.error(f"Failed to convert {layout_file}: {e}")


def get_available_layouts(layouts_dir: Path) -> Dict[str, Path]:
    """Get all available layout files in the directory

    Args:
        layouts_dir: Directory containing layout files

    Returns:
        Dictionary mapping layout names to file paths
    """
    layouts = {}

    # First look for YAML files (preferred format)
    for file_path in layouts_dir.glob("*.yaml"):
        layout_name = file_path.stem
        layouts[layout_name] = file_path

    # Then look for legacy layout files that haven't been converted yet
    for file_path in layouts_dir.glob("*.layout"):
        # Only add if a YAML version doesn't already exist
        layout_name = file_path.stem
        if layout_name not in layouts:
            layouts[layout_name] = file_path

    return layouts


if __name__ == "__main__":
    # Simple CLI for converting layouts
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert keyboard layouts to YAML format"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/keyLayouts"),
        help="Directory containing layout files",
    )
    parser.add_argument(
        "--convert-all", action="store_true", help="Convert all legacy layout files"
    )
    parser.add_argument(
        "--regenerate-comfort",
        action="store_true",
        help="Regenerate comfort matrices for all layouts",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.convert_all:
        convert_all_layouts(args.dir)

    if args.regenerate_comfort:
        for layout_path in args.dir.glob("*.yaml"):
            try:
                load_keyboard_layout(layout_path, force_regenerate_comfort=True)
            except Exception as e:
                logger.error(f"Failed to regenerate comfort for {layout_path}: {e}")
