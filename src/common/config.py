"""Configuration handler"""

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .metrics import (
    AssignmentMetricType,
    SetMetricType,
    StandaloneCombinationType,
    StandaloneMovementType,
)


@dataclass
class StandaloneWeights:
    """Weights for standalone metrics"""

    movement_weights: Dict[StandaloneMovementType, float] = field(default_factory=dict)
    combination_weights: Dict[StandaloneCombinationType, float] = field(
        default_factory=dict
    )


@dataclass
class AssignmentWeights:
    """Weights for assignment metrics"""

    metric_weights: Dict[AssignmentMetricType, float] = field(default_factory=dict)


@dataclass
class SetWeights:
    """Weights for set metrics"""

    metric_weights: Dict[SetMetricType, float] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    """Configuration parameters for chord generation"""

    # Required parameters
    keylayout_type: str
    keylayout_csv_file: Path
    max_letters: int
    min_letters: int
    output_type: str

    # Weights
    standalone_weights: StandaloneWeights
    assignment_weights: AssignmentWeights
    set_weights: SetWeights

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> "GeneratorConfig":
        """Load configuration from file"""
        config = configparser.ConfigParser(inline_comment_prefixes="#")

        if config_path is None:
            config_path = Path("generator.config")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config.read(config_path)

        # Required fields
        required = config["REQUIRED"]
        standalone = config["STANDALONE_WEIGHTS"]
        assignment = config["ASSIGNMENT_WEIGHTS"]
        set_weights = config["SET_WEIGHTS"]

        # Load movement weights using enum
        movement_weights = {
            movement_type: float(standalone[f"{movement_type.name}_WEIGHT"])
            for movement_type in StandaloneMovementType
        }

        # Load combination weights using enum
        combination_weights = {
            combination_type: float(standalone[f"{combination_type.name}_WEIGHT"])
            for combination_type in StandaloneCombinationType
        }

        # Load assignment weights using enum
        assignment_weights = {
            metric_type: float(assignment[f"{metric_type.name}_WEIGHT"])
            for metric_type in AssignmentMetricType
        }

        # Load set weights using enum
        set_metric_weights = {
            metric_type: float(set_weights[f"{metric_type.name}_WEIGHT"])
            for metric_type in SetMetricType
        }

        return cls(
            keylayout_type=required["KEYLAYOUT_TYPE"],
            keylayout_csv_file=Path(required["KEYLAYOUT_CSV_FILE"]),
            max_letters=int(required["MAX_LETTERS"]),
            min_letters=int(required["MIN_LETTERS"]),
            output_type=required["OUTPUT_TYPE"],
            standalone_weights=StandaloneWeights(
                movement_weights=movement_weights,
                combination_weights=combination_weights,
            ),
            assignment_weights=AssignmentWeights(metric_weights=assignment_weights),
            set_weights=SetWeights(metric_weights=set_metric_weights),
        )

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

        # Validate that all enum values have weights
        for movement_type in StandaloneMovementType:
            if movement_type not in self.standalone_weights.movement_weights:
                raise ValueError(f"Missing weight for movement type: {movement_type}")

        for combination_type in StandaloneCombinationType:
            if combination_type not in self.standalone_weights.combination_weights:
                raise ValueError(
                    f"Missing weight for combination type: {combination_type}"
                )

        for metric_type in AssignmentMetricType:
            if metric_type not in self.assignment_weights.metric_weights:
                raise ValueError(f"Missing weight for assignment metric: {metric_type}")

        for metric_type in SetMetricType:
            if metric_type not in self.set_weights.metric_weights:
                raise ValueError(f"Missing weight for set metric: {metric_type}")
