"""Layout loader for keyboard configurations

This module handles loading and processing of YAML-based keyboard layouts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.common.shared_types import Finger, KeyPosition

logger = logging.getLogger(__name__)


def load_layout_from_yaml(layout_file: Path) -> Dict:
    """Load keyboard layout from YAML file

    Args:
        layout_file: Path to the YAML layout file

    Returns:
        Dict containing the layout data

    Raises:
        FileNotFoundError: If layout file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not layout_file.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_file}")

    with open(layout_file, "r", encoding="utf-8") as f:
        layout_data = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["positions", "fingers", "usage_cost"]
    missing_sections = [
        section for section in required_sections if section not in layout_data
    ]

    if missing_sections:
        raise ValueError(f"Layout file missing required sections: {missing_sections}")

    # Validate usage_cost matrix has positive values
    usage_cost = layout_data["usage_cost"]
    invalid_values = [key for key, value in usage_cost.items() if value <= 0]

    if invalid_values:
        raise ValueError(
            f"Layout has non-positive usage_cost values for keys: {invalid_values}"
        )

    # Ensure 'unknown' key exists
    if "unknown" not in usage_cost:
        raise ValueError("Layout missing required 'unknown' entry in usage_cost matrix")

    return layout_data


def save_layout_to_yaml(layout_data: Dict, layout_file: Path) -> None:
    """Save keyboard layout to YAML file

    Args:
        layout_data: The layout data dictionary to save
        layout_file: Path where to save the layout file
    """
    # Create parent directory if it doesn't exist
    layout_file.parent.mkdir(parents=True, exist_ok=True)

    with open(layout_file, "w", encoding="utf-8") as f:
        yaml.dump(layout_data, f, default_flow_style=False, sort_keys=False)


def create_key_positions(layout_data: Dict) -> Dict[str, KeyPosition]:
    """Create KeyPosition objects from layout data

    Args:
        layout_data: Dictionary containing layout information

    Returns:
        Dictionary mapping keys to KeyPosition objects
    """
    # Extract position and finger data
    positions = layout_data.get("positions", {})
    fingers = layout_data.get("fingers", {})

    # Create key positions
    key_positions = {}
    for key, pos in positions.items():
        row, col = pos
        finger_name = fingers.get(key)

        if not finger_name:
            logger.warning(
                f"Key '{key}' has position but no finger assignment, skipping"
            )
            continue

        # Get finger enum
        try:
            finger = Finger[finger_name]
        except KeyError:
            logger.error(f"Invalid finger name '{finger_name}' for key '{key}'")
            continue

        # Find adjacent finger assignments
        finger_to_left = None
        finger_to_right = None

        # Check adjacent keys in the same row to infer finger relationships
        for other_key, other_pos in positions.items():
            other_row, other_col = other_pos
            other_finger_name = fingers.get(other_key)

            if other_finger_name and other_row == row:
                if other_col == col - 1:  # Key to the left
                    try:
                        finger_to_left = Finger[other_finger_name]
                    except KeyError:
                        logger.warning(
                            f"Invalid finger name '{other_finger_name}' for adjacent key"
                        )
                elif other_col == col + 1:  # Key to the right
                    try:
                        finger_to_right = Finger[other_finger_name]
                    except KeyError:
                        logger.warning(
                            f"Invalid finger name '{other_finger_name}' for adjacent key"
                        )

        # Create KeyPosition object
        key_positions[key] = KeyPosition(
            finger=finger,
            vertical_distance_to_resting_position=row,
            horizontal_distance_to_resting_position=col,
            finger_to_left=finger_to_left,
            finger_to_right=finger_to_right,
        )

    return key_positions


def load_keyboard_layout(layout_file: Path) -> Dict:
    """Load keyboard layout with all data needed for analysis

    Args:
        layout_file: Path to the layout file

    Returns:
        Dictionary with complete layout data including KeyPosition objects

    Raises:
        FileNotFoundError: If layout file doesn't exist
        ValueError: If layout data is invalid
    """
    # Load YAML file
    layout_data = load_layout_from_yaml(layout_file)

    # Create KeyPosition objects
    key_positions = create_key_positions(layout_data)

    # Return complete layout data
    return {
        "name": layout_data.get("name", layout_file.stem),
        "type": layout_data.get("type", "unknown"),
        "description": layout_data.get("description", ""),
        "positions": layout_data.get("positions", {}),
        "usage_cost": layout_data.get("usage_cost", {}),
        "fingers": layout_data.get("fingers", {}),
        "key_positions": key_positions,
    }


def get_available_layouts(layouts_dir: Path) -> Dict[str, Path]:
    """Get all available layout files in the directory

    Args:
        layouts_dir: Directory containing layout files

    Returns:
        Dictionary mapping layout names to file paths
    """
    layouts = {}

    # Look for YAML files only
    for file_path in layouts_dir.glob("*.yaml"):
        layout_name = file_path.stem
        layouts[layout_name] = file_path

    if not layouts:
        logger.warning(f"No YAML layout files found in {layouts_dir}")

    return layouts


def validate_layout_data(layout_data: Dict) -> List[str]:
    """Validate layout data structure and return list of validation errors

    Args:
        layout_data: The layout data to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required top-level keys
    required_keys = ["name", "positions", "fingers", "usage_cost"]
    for key in required_keys:
        if key not in layout_data:
            errors.append(f"Missing required key: {key}")

    if "positions" in layout_data and "fingers" in layout_data:
        positions = layout_data["positions"]
        fingers = layout_data["fingers"]

        # Check that all positions have corresponding finger assignments
        for key in positions:
            if key not in fingers:
                errors.append(f"Key '{key}' has position but no finger assignment")

        # Check that all finger assignments have corresponding positions
        for key in fingers:
            if key not in positions:
                errors.append(f"Key '{key}' has finger assignment but no position")

        # Validate finger names
        valid_fingers = {finger.name for finger in Finger}
        for key, finger_name in fingers.items():
            if finger_name not in valid_fingers:
                errors.append(f"Invalid finger name '{finger_name}' for key '{key}'")

    if "usage_cost" in layout_data:
        usage_cost = layout_data["usage_cost"]

        # Check that usage costs are positive
        for key, cost in usage_cost.items():
            if not isinstance(cost, (int, float)) or cost <= 0:
                errors.append(
                    f"Usage cost for key '{key}' must be positive, got {cost}"
                )

        # Check for required 'unknown' key
        if "unknown" not in usage_cost:
            errors.append("Missing required 'unknown' entry in usage_cost matrix")

    return errors


if __name__ == "__main__":
    # Simple CLI for layout operations
    import argparse

    parser = argparse.ArgumentParser(description="Layout file operations")
    parser.add_argument("--validate", type=Path, help="Validate a specific layout file")
    parser.add_argument(
        "--list",
        type=Path,
        default=Path("data/keyLayouts"),
        help="List available layouts in directory",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.validate:
        try:
            layout_data = load_keyboard_layout(args.validate)
            errors = validate_layout_data(layout_data)

            if errors:
                logger.error(f"Validation failed for {args.validate}:")
                for error in errors:
                    logger.error(f"  - {error}")
            else:
                logger.info(f"Layout {args.validate} is valid")
                logger.info(f"  Name: {layout_data['name']}")
                logger.info(f"  Type: {layout_data['type']}")
                logger.info(f"  Keys: {len(layout_data['positions'])}")
        except Exception as e:
            logger.error(f"Error loading layout {args.validate}: {e}")

    else:
        layouts = get_available_layouts(args.list)
        logger.info(f"Found {len(layouts)} layout(s) in {args.list}:")
        for name, path in sorted(layouts.items()):
            logger.info(f"  - {name} ({path.name})")
