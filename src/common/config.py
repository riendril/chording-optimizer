"""Configuration handler

This module provides configuration management for the chord generator, including:
- General parameters (keyboard format, input/output files)
- Debug settings (logging level, debug output)
- Performance benchmarking options
- Metric weights (standalone, assignment, and set metrics)
"""

import configparser
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from src.common.shared_types import (
    AssignmentMetricType,
    OutputFormat,
    SetMetricType,
    StandaloneMetricType,
)


class LogLevel(Enum):
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


@dataclass
class DebugOptions:
    """Debug and logging options"""

    enabled: bool
    log_level: LogLevel
    log_file: Optional[Path]
    print_cost_details: bool  # Print detailed info for each generated chord
    save_intermediate_results: bool  # Save intermediate processing results


@dataclass
class BenchmarkOptions:
    """Performance benchmarking configuration"""

    enabled: bool
    include_memory_stats: bool
    include_timing_stats: bool
    track_generation_phases: bool
    track_metric_types: bool
    sample_interval: int


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
    keylayout_file: Path
    max_letters: int
    min_letters: int
    output_format: OutputFormat

    # Debug and benchmark options
    debug: DebugOptions
    benchmark: BenchmarkOptions

    # Weights
    standalone_weights: StandaloneWeights
    assignment_weights: AssignmentWeights
    set_weights: SetWeights

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> "GeneratorConfig":
        """Load configuration from file

        Args:
            config_path: Optional path to config file. If None, uses default path.

        Returns:
            Loaded configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config values are invalid
        """
        config = configparser.ConfigParser(inline_comment_prefixes="#")

        if config_path is None:
            config_path = Path("data/input/generator.config")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config.read(config_path)

        # Load sections
        general_section = config["GENERAL"]
        debug_section = config["DEBUG"]
        benchmark_section = config["BENCHMARK"]
        standalone_weights_section = config["STANDALONE_WEIGHTS"]
        assignment_weights_section = config["ASSIGNMENT_WEIGHTS"]
        set_weights_section = config["SET_WEIGHTS"]

        # Parse debug config
        debug_config = DebugOptions(
            enabled=bool(debug_section["ENABLED"]),
            log_level=LogLevel[debug_section["LOG_LEVEL"]],
            log_file=Path(debug_section["LOG_FILE"]),
            print_cost_details=bool(debug_section["PRINT_COST_DETAILS"]),
            save_intermediate_results=bool(debug_section["SAVE_INTERMEDIATE_RESULTS"]),
        )

        # Parse benchmark config
        benchmark_config = BenchmarkOptions(
            enabled=bool(benchmark_section["ENABLED"]),
            include_memory_stats=bool(benchmark_section["INCLUDE_MEMORY_STATS"]),
            include_timing_stats=bool(benchmark_section["INCLUDE_TIMING_STATS"]),
            track_generation_phases=bool(benchmark_section["TRACK_GENERATION_PHASES"]),
            track_metric_types=bool(benchmark_section["TRACK_METRIC_TYPES"]),
            sample_interval=int(benchmark_section["SAMPLE_INTERVAL"]),
        )

        # Load metric weights
        standalone_weights = {
            metric_type: float(standalone_weights_section[f"{metric_type.name}_WEIGHT"])
            for metric_type in StandaloneMetricType
        }

        assignment_weights = {
            metric_type: float(assignment_weights_section[f"{metric_type.name}_WEIGHT"])
            for metric_type in AssignmentMetricType
        }

        set_weights = {
            metric_type: float(set_weights_section[f"{metric_type.name}_WEIGHT"])
            for metric_type in SetMetricType
        }

        return cls(
            keylayout_file=Path(general_section["KEYLAYOUT_FILE"]),
            max_letters=int(general_section["MAX_LETTERS"]),
            min_letters=int(general_section["MIN_LETTERS"]),
            output_format=OutputFormat[general_section["OUTPUT_FORMAT"]],
            debug=debug_config,
            benchmark=benchmark_config,
            standalone_weights=StandaloneWeights(weights=standalone_weights),
            assignment_weights=AssignmentWeights(weights=assignment_weights),
            set_weights=SetWeights(weights=set_weights),
        )

    def validate(self) -> None:
        """Validate configuration values

        Raises:
            ValueError: If any configuration values are invalid
        """
        if self.min_letters < 1:
            raise ValueError("MIN_LETTERS must be at least 1")
        if self.max_letters < self.min_letters:
            raise ValueError("MAX_LETTERS must be greater than or equal to MIN_LETTERS")
        if not isinstance(self.output_format, OutputFormat):
            raise ValueError(f"Unsupported OUTPUT_FORMAT: {self.output_format}")

        # Validate debug config
        if self.debug.enabled and self.debug.log_file:
            if not self.debug.log_file.parent.exists():
                raise ValueError(
                    f"Log file directory does not exist: {self.debug.log_file.parent}"
                )

        # Validate layout file exists
        if not self.keylayout_file.exists():
            raise ValueError(f"Layout file does not exist: {self.keylayout_file}")

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

    def setup_logging(self) -> None:
        """Configure logging based on debug settings"""
        if not self.debug.enabled:
            return

        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        if self.debug.log_file:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(self.debug.log_file),
                    logging.StreamHandler(),
                ],
            )
        else:
            logging.basicConfig(level=logging.INFO, format=log_format)
