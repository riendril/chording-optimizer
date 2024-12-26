"""Configuration handler for chord generator"""

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeneratorConfig:
    """Configuration parameters for chord generation"""

    # Input files
    keylayout_type: str
    keylayout_csv_file: str
    corpus_json_file: str

    # Rules
    max_letters: int
    min_letters: int

    # Weights
    chord_length_weight: float
    fallback_penalty: float
    first_letter_weight: float
    second_letter_weight: float
    last_letter_weight: float

    # Output
    output_type: str

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

        # Validate all required fields are present
        required_fields = [
            "KEYLAYOUT_TYPE",
            "KEYLAYOUT_CSV_FILE",
            "CORPUS_JSON_FILE",
            "MAX_LETTERS",
            "MIN_LETTERS",
            "CHORD_LENGTH_WEIGHT",
            "FALLBACK_PENALTY",
            "FIRST_LETTER_WEIGHT",
            "SECOND_LETTER_WEIGHT",
            "LAST_LETTER_WEIGHT",
            "OUTPUT_TYPE",
        ]

        missing_fields = [
            field for field in required_fields if field not in main_config
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in config: {', '.join(missing_fields)}"
            )

        return cls(
            keylayout_type=main_config["KEYLAYOUT_TYPE"],
            keylayout_csv_file=main_config["KEYLAYOUT_CSV_FILE"],
            corpus_json_file=main_config["CORPUS_JSON_FILE"],
            max_letters=int(main_config["MAX_LETTERS"]),
            min_letters=int(main_config["MIN_LETTERS"]),
            chord_length_weight=float(main_config["CHORD_LENGTH_WEIGHT"]),
            fallback_penalty=float(main_config["FALLBACK_PENALTY"]),
            first_letter_weight=float(main_config["FIRST_LETTER_WEIGHT"]),
            second_letter_weight=float(main_config["SECOND_LETTER_WEIGHT"]),
            last_letter_weight=float(main_config["LAST_LETTER_WEIGHT"]),
            output_type=main_config["OUTPUT_TYPE"],
        )

    def validate(self) -> None:
        """Validate configuration values"""
        if self.min_letters < 1:
            raise ValueError("MIN_LETTERS must be at least 1")
        if self.max_letters < self.min_letters:
            raise ValueError("MAX_LETTERS must be greater than or equal to MIN_LETTERS")
        if self.keylayout_type not in ["ortholinear"]:
            raise ValueError("Unsupported KEYLAYOUT_TYPE")
        if self.output_type not in ["visual"]:
            raise ValueError("Unsupported OUTPUT_TYPE")
