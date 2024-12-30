"""Configuration handler"""

import configparser
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional

from src.common.layout import KeyboardFormat
from src.common.shared_types import (
    AssignmentMetricType,
    SetMetricType,
    StandaloneMetricType,
)


class OutputFormat(Enum):
    """Enum for the currently supported output formats"""

    STANDARD_JSON = auto()


@dataclass
class StandaloneWeights:
    """Weights for standalone metrics"""

    weights: Dict[StandaloneMetricType, float] = field(default_factory=dict)


@dataclass
class AssignmentWeights:
    """Weights for assignment metrics"""

    weights: Dict[AssignmentMetricType, float] = field(default_factory=dict)


@dataclass
class SetWeights:
    """Weights for set metrics"""

    weights: Dict[SetMetricType, float] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    """Configuration parameters for chord generation"""

    # General parameters
    keyboard_format: KeyboardFormat
    keylayout_csv_file: Path
    max_letters: int
    min_letters: int
    output_format: OutputFormat

    # Weights
    standalone_weights: StandaloneWeights
    assignment_weights: AssignmentWeights
    set_weights: SetWeights

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> "GeneratorConfig":
        """Load configuration from file"""
        config = configparser.ConfigParser(inline_comment_prefixes="#")

        if config_path is None:
            config_path = Path("data/input/generator.config")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config.read(config_path)

        general_section = config["GENERAL"]
        standalone_weights_section = config["STANDALONE_WEIGHTS"]
        assignment_weights_section = config["ASSIGNMENT_WEIGHTS"]
        set_weights_section = config["SET_WEIGHTS"]

        # Load standalone weights using enum
        standalone_weights = {
            metric_type: float(standalone_weights_section[f"{metric_type.name}_WEIGHT"])
            for metric_type in StandaloneMetricType
        }

        # Load assignment weights using enum
        assignment_weights = {
            metric_type: float(assignment_weights_section[f"{metric_type.name}_WEIGHT"])
            for metric_type in AssignmentMetricType
        }

        # Load set weights using enum
        set_metric_weights = {
            metric_type: float(set_weights_section[f"{metric_type.name}_WEIGHT"])
            for metric_type in SetMetricType
        }

        return cls(
            keyboard_format=KeyboardFormat(general_section["KEYBOARD_FORMAT"]),
            keylayout_csv_file=Path(general_section["KEYLAYOUT_CSV_FILE"]),
            max_letters=int(general_section["MAX_LETTERS"]),
            min_letters=int(general_section["MIN_LETTERS"]),
            output_format=OutputFormat(general_section["OUTPUT_FORMAT"]),
            standalone_weights=StandaloneWeights(weights=standalone_weights),
            assignment_weights=AssignmentWeights(weights=assignment_weights),
            set_weights=SetWeights(weights=set_metric_weights),
        )

    def validate(self) -> None:
        """Validate configuration values"""
        if self.min_letters < 1:
            raise ValueError("MIN_LETTERS must be at least 1")
        if self.max_letters < self.min_letters:
            raise ValueError("MAX_LETTERS must be greater than or equal to MIN_LETTERS")
        if self.keyboard_format not in ["matrix"]:
            raise ValueError("Unsupported KEYBOARD_FORMAT")
        if self.output_format not in ["visual"]:
            raise ValueError("Unsupported OUTPUT_FORMAT")

        # Validate that all enum values have weights
        for metric_type in StandaloneMetricType:
            if metric_type not in self.standalone_weights.weights:
                raise ValueError(f"Missing weight for standalone metric: {metric_type}")

        for metric_type in AssignmentMetricType:
            if metric_type not in self.assignment_weights.weights:
                raise ValueError(f"Missing weight for assignment metric: {metric_type}")

        for metric_type in SetMetricType:
            if metric_type not in self.set_weights.weights:
                raise ValueError(f"Missing weight for set metric: {metric_type}")
