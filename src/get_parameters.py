"""Configuration handler for chord generator"""

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GeneratorConfig:
    """Configuration parameters for chord generation"""

    # Required parameters (no defaults)
    keylayout_type: str
    keylayout_csv_file: str
    corpus_json_file: str
    max_letters: int
    min_letters: int
    output_type: str

    # Optional parameters with default 1
    # Letter weights
    additional_letter_weight: float = field(default=1.0)
    fallback_letter_weight: float = field(default=1.0)
    first_letter_unmatched_weight: float = field(default=1.0)
    second_letter_unmatched_weight: float = field(default=1.0)
    last_letter_unmatched_weight: float = field(default=1.0)

    # Finger movement weights
    vertical_stretch_weight: float = field(default=1.0)
    vertical_pinch_weight: float = field(default=1.0)
    horizontal_stretch_weight: float = field(default=1.0)
    horizontal_pinch_weight: float = field(default=1.0)
    diagonal_stretch_weight: float = field(default=1.0)
    diagonal_pinch_weight: float = field(default=1.0)

    # Finger combination weights
    same_finger_double_weight: float = field(default=1.0)
    same_finger_triple_weight: float = field(default=1.0)

    # Awkward combination weights
    pinky_ring_stretch_weight: float = field(default=1.0)
    ring_middle_scissor_weight: float = field(default=1.0)
    middle_index_stretch_weight: float = field(default=1.0)

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> "GeneratorConfig":
        """Load configuration from file"""
        config = configparser.ConfigParser(inline_comment_prefixes="#")

        # Use generator.config as default if no path provided
        if config_path is None:
            config_path = Path("generator.config")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config.read(config_path)

        # Get main section
        if "DEFAULT" not in config:
            raise ValueError("Invalid config file: missing DEFAULT section")

        main_config = config["DEFAULT"]

        # Required fields
        required_fields = [
            "KEYLAYOUT_TYPE",
            "KEYLAYOUT_CSV_FILE",
            "MAX_LETTERS",
            "MIN_LETTERS",
            "OUTPUT_TYPE",
        ]

        missing_fields = [
            field for field in required_fields if field not in main_config
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in config: {', '.join(missing_fields)}"
            )

        # Initialize with required fields
        config_dict = {
            "keylayout_type": main_config["KEYLAYOUT_TYPE"],
            "keylayout_csv_file": main_config["KEYLAYOUT_CSV_FILE"],
            "corpus_json_file": main_config["CORPUS_JSON_FILE"],
            "max_letters": int(main_config["MAX_LETTERS"]),
            "min_letters": int(main_config["MIN_LETTERS"]),
            "output_type": main_config["OUTPUT_TYPE"],
        }

        # Optional weight fields with their config keys
        weights = {
            "additional_letter_weight": "ADDITIONAL_LETTER_WEIGHT",
            "fallback_letter_weight": "FALLBACK_LETTER_WEIGHT",
            "first_letter_unmatched_weight": "FIRST_LETTER_UNMATCHED_WEIGHT",
            "second_letter_unmatched_weight": "SECOND_LETTER_UNMATCHED_WEIGHT",
            "last_letter_unmatched_weight": "LAST_LETTER_UNMATCHED_WEIGHT",
            "vertical_stretch_weight": "VERTICAL_STRETCH_WEIGHT",
            "vertical_pinch_weight": "VERTICAL_PINCH_WEIGHT",
            "horizontal_stretch_weight": "HORIZONTAL_STRETCH_WEIGHT",
            "horizontal_pinch_weight": "HORIZONTAL_PINCH_WEIGHT",
            "diagonal_stretch_weight": "DIAGONAL_STRETCH_WEIGHT",
            "diagonal_pinch_weight": "DIAGONAL_PINCH_WEIGHT",
            "same_finger_double_weight": "SAME_FINGER_DOUBLE_WEIGHT",
            "same_finger_triple_weight": "SAME_FINGER_TRIPLE_WEIGHT",
            "pinky_ring_stretch_weight": "PINKY_RING_STRETCH_WEIGHT",
            "ring_middle_scissor_weight": "RING_MIDDLE_SCISSOR_WEIGHT",
            "middle_index_stretch_weight": "MIDDLE_INDEX_STRETCH_WEIGHT",
        }

        # Load optional weights if present
        for weight, value in weights.items():
            if value in main_config:
                config_dict[weight] = float(main_config[value])

        return cls(**config_dict)

    def validate(self) -> None:
        """Validate configuration values"""
        if self.min_letters < 1:
            raise ValueError("MIN_LETTERS must be at least 1")
        if self.max_letters < self.min_letters:
            raise ValueError("MAX_LETTERS must be greater than or equal to MIN_LETTERS")
        if self.keylayout_type not in ["matrix"]:
            raise ValueError("Unsupported KEYLAYOUT_TYPE")
        if self.output_type not in ["visual"]:
            raise ValueError("Unsupported OUTPUT_TYPE")
