"""Configuration handler

This module provides YAML-based configuration management for the chord generator, including:
- General parameters (keyboard layout, input/output paths)
- Debug settings (logging level, debug output)
- Performance benchmarking options
- Metric weights (standalone, assignment, and set metrics)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.common.shared_types import (
    AssignmentMetricType,
    Finger,
    SetMetricType,
    StandaloneMetricType,
)


class LogLevel(Enum):
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


@dataclass
class Paths:
    """Path configuration for input and output files"""

    # Base directories
    base_dir: Path = Path(".")
    data_dir: Path = Path("data")

    # Input directories
    key_layouts_dir: Path = Path("data/keyLayouts")
    corpuses_dir: Path = Path("data/corpuses")
    tokens_dir: Path = Path("data/tokens")

    # Output directories
    chords_dir: Path = Path("data/chords")
    debug_dir: Path = Path("data/debug")
    results_dir: Path = Path("data/results")
    cache_dir: Path = Path("data/cache")  # Added for corpus generator

    # Default files
    default_layout_file: Path = Path("data/keyLayouts/Nordrassil_Ergonomic.yaml")
    default_corpus_file: Path = Path("data/corpuses/brown.txt")
    default_tokens_file: Path = Path("data/tokens/MonkeyType_english_1k.json")

    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [
            self.data_dir,
            self.key_layouts_dir,
            self.corpuses_dir,
            self.tokens_dir,
            self.chords_dir,
            self.debug_dir,
            self.results_dir,
            self.cache_dir,  # Added for corpus generator
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def get_layout_path(self, layout_name: Optional[str] = None) -> Path:
        """Get path to a specific layout file or default"""
        if layout_name:
            # Check for .yaml extension first
            yaml_path = self.key_layouts_dir / f"{layout_name}.yaml"
            if yaml_path.exists():
                return yaml_path

            # Try .layout extension for backward compatibility
            layout_path = self.key_layouts_dir / f"{layout_name}.layout"
            if layout_path.exists():
                return layout_path

        return self.default_layout_file

    def get_corpus_path(self, corpus_name: Optional[str] = None) -> Path:
        """Get path to a specific corpus file or default"""
        if corpus_name:
            return self.corpuses_dir / f"{corpus_name}.txt"
        return self.default_corpus_file

    def get_tokens_path(self, tokens_name: Optional[str] = None) -> Path:
        """Get path to a specific tokens file or default"""
        if tokens_name:
            return self.tokens_dir / f"{tokens_name}.json"
        return self.default_tokens_file


@dataclass
class DebugOptions:
    """Debug and logging options"""

    enabled: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[Path] = Path("data/debug/chord_generator.log")
    print_cost_details: bool = False
    save_intermediate_results: bool = False


@dataclass
class BenchmarkOptions:
    """Performance benchmarking configuration"""

    enabled: bool = False
    track_individual_metrics: bool = False
    visual_update_interval: int = 100


@dataclass
class ChordGeneration:
    """Chord generation parameters"""

    min_letter_count: int = 2
    max_letter_count: int = 6
    allow_non_adjacent_keys: bool = True


@dataclass
class CorpusGenerationConfig:
    """Corpus generation parameters"""

    sample_size: int = 1000
    min_length: int = 50
    max_length: int = 500
    total_corpus_size: int = 5000000
    categories: Dict[str, float] = field(
        default_factory=lambda: {
            "forum": 0.3,
            "subtitles": 0.2,
            "scientific": 0.15,
            "programming": 0.2,
            "general": 0.15,
        }
    )
    api_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "opensubtitles": "YOUR_OPENSUBTITLES_API_KEY",
            "github": "YOUR_GITHUB_API_KEY",
        }
    )


@dataclass
class StandaloneWeights:
    """Weights for standalone metrics"""

    weights: Dict[StandaloneMetricType, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default weights if not provided"""
        defaults = {
            StandaloneMetricType.CHORD_LENGTH: 1.0,
            StandaloneMetricType.HORIZONTAL_PINCH: 1.2,
            StandaloneMetricType.HORIZONTAL_STRETCH: 1.3,
            StandaloneMetricType.VERTICAL_PINCH: 1.2,
            StandaloneMetricType.VERTICAL_STRETCH: 1.4,
            StandaloneMetricType.DIAGONAL_DOWNWARD_PINCH: 1.3,
            StandaloneMetricType.DIAGONAL_DOWNWARD_STRETCH: 1.5,
            StandaloneMetricType.DIAGONAL_UPWARD_PINCH: 1.3,
            StandaloneMetricType.DIAGONAL_UPWARD_STRETCH: 1.5,
            StandaloneMetricType.SAME_FINGER_DOUBLE_ADJACENT: 1.5,
            StandaloneMetricType.SAME_FINGER_DOUBLE_GAP: 2.0,
            StandaloneMetricType.SAME_FINGER_TRIPLE: 3.0,
            StandaloneMetricType.FULL_SCISSOR_DOUBLE: 1.7,
            StandaloneMetricType.FULL_SCISSOR_TRIPLE: 2.5,
            StandaloneMetricType.FULL_SCISSOR_QUADRUPLE: 3.0,
            StandaloneMetricType.FULL_SCISSOR_QUINTUPLE: 4.0,
            StandaloneMetricType.HALF_SCISSOR_DOUBLE: 1.5,
            StandaloneMetricType.HORIZONTAL_STRETCH_DOUBLE: 1.6,
            StandaloneMetricType.PINKY_RING_SCISSOR: 2.0,
            StandaloneMetricType.RING_INDEX_SCISSOR: 1.8,
        }

        for metric, weight in defaults.items():
            if metric not in self.weights:
                self.weights[metric] = weight


@dataclass
class AssignmentWeights:
    """Weights for assignment metrics"""

    weights: Dict[AssignmentMetricType, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default weights if not provided"""
        defaults = {
            AssignmentMetricType.FIRST_LETTER_UNMATCHED: 1.5,
            AssignmentMetricType.SECOND_LETTER_UNMATCHED: 1.2,
            AssignmentMetricType.LAST_LETTER_UNMATCHED: 1.3,
            AssignmentMetricType.PHONETIC_DISSIMILARITY: 1.1,
            AssignmentMetricType.EXTRA_LETTER: 1.2,
        }

        for metric, weight in defaults.items():
            if metric not in self.weights:
                self.weights[metric] = weight


@dataclass
class SetWeights:
    """Weights for set metrics"""

    weights: Dict[SetMetricType, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default weights if not provided"""
        defaults = {
            SetMetricType.FINGER_UTILIZATION: 1.0,
            SetMetricType.HAND_UTILIZATION: 1.0,
            SetMetricType.CHORD_PATTERN_CONSISTENCY: 1.0,
        }

        for metric, weight in defaults.items():
            if metric not in self.weights:
                self.weights[metric] = weight


@dataclass
class TokenAnalysisConfig:
    """Configuration for token analysis"""

    min_token_length: int = 1
    max_token_length: int = 10
    top_n_tokens: int = 1000
    include_characters: bool = True
    include_character_ngrams: bool = True
    include_words: bool = True
    include_word_ngrams: bool = True
    use_parallel_processing: bool = True


@dataclass
class ChordAssignmentConfig:
    """Configuration for chord assignment algorithms"""

    algorithm: str = "algorithm1"
    first_letter_unmatched_weight: float = 1.5
    second_letter_unmatched_weight: float = 1.2
    last_letter_unmatched_weight: float = 1.3
    additional_letter_weight: float = 1.2
    fallback_letter_weight: float = 1.5
    vertical_stretch_weight: float = 1.4
    vertical_pinch_weight: float = 1.2
    horizontal_stretch_weight: float = 1.3
    horizontal_pinch_weight: float = 1.2
    diagonal_stretch_weight: float = 1.5
    diagonal_pinch_weight: float = 1.3
    same_finger_double_weight: float = 1.5
    same_finger_triple_weight: float = 3.0
    pinky_ring_stretch_weight: float = 2.0
    ring_middle_scissor_weight: float = 1.8
    middle_index_stretch_weight: float = 1.6


@dataclass
class GeneratorConfig:
    """Main configuration for the chord generator"""

    # Core components
    paths: Paths = field(default_factory=Paths)
    debug: DebugOptions = field(default_factory=DebugOptions)
    benchmark: BenchmarkOptions = field(default_factory=BenchmarkOptions)

    # Generation parameters
    chord_generation: ChordGeneration = field(default_factory=ChordGeneration)
    corpus_generation: CorpusGenerationConfig = field(
        default_factory=CorpusGenerationConfig
    )

    # Weights
    standalone_weights: StandaloneWeights = field(default_factory=StandaloneWeights)
    assignment_weights: AssignmentWeights = field(default_factory=AssignmentWeights)
    set_weights: SetWeights = field(default_factory=SetWeights)

    # Module-specific configs
    token_analysis: TokenAnalysisConfig = field(default_factory=TokenAnalysisConfig)
    chord_assignment: ChordAssignmentConfig = field(
        default_factory=ChordAssignmentConfig
    )

    # Current active settings
    active_layout: str = "norddrassil_ergonomic"
    active_corpus: str = "brown"
    active_tokens: str = "MonkeyType_english_1k"

    @classmethod
    def load_config(
        cls, config_path: Optional[Union[str, Path]] = None
    ) -> "GeneratorConfig":
        """Load configuration from YAML file

        Args:
            config_path: Optional path to config file. If None, uses default path.

        Returns:
            Loaded configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config values are invalid
        """
        if config_path is None:
            config_path = Path("data/config.yaml")

        config_path = Path(config_path)

        if not config_path.exists():
            # If no config file exists, create default config
            config = cls()
            config.save_config(config_path)
            return config

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Convert YAML data to GeneratorConfig
        return cls._from_dict(config_data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "GeneratorConfig":
        """Create config from dictionary"""
        config = cls()

        # Parse paths
        if "paths" in data:
            for key, value in data["paths"].items():
                if hasattr(config.paths, key):
                    setattr(config.paths, key, Path(value))

        # Parse debug options
        if "debug" in data:
            debug_data = data["debug"]
            if "enabled" in debug_data:
                config.debug.enabled = debug_data["enabled"]
            if "log_level" in debug_data:
                config.debug.log_level = LogLevel[debug_data["log_level"]]
            if "log_file" in debug_data:
                config.debug.log_file = Path(debug_data["log_file"])
            if "print_cost_details" in debug_data:
                config.debug.print_cost_details = debug_data["print_cost_details"]
            if "save_intermediate_results" in debug_data:
                config.debug.save_intermediate_results = debug_data[
                    "save_intermediate_results"
                ]

        # Parse benchmark options
        if "benchmark" in data:
            bench_data = data["benchmark"]
            if "enabled" in bench_data:
                config.benchmark.enabled = bench_data["enabled"]
            if "track_individual_metrics" in bench_data:
                config.benchmark.track_individual_metrics = bench_data[
                    "track_individual_metrics"
                ]
            if "visual_update_interval" in bench_data:
                config.benchmark.visual_update_interval = bench_data[
                    "visual_update_interval"
                ]

        # Parse chord generation
        if "chord_generation" in data:
            gen_data = data["chord_generation"]
            if "min_letter_count" in gen_data:
                config.chord_generation.min_letter_count = gen_data["min_letter_count"]
            if "max_letter_count" in gen_data:
                config.chord_generation.max_letter_count = gen_data["max_letter_count"]
            if "allow_non_adjacent_keys" in gen_data:
                config.chord_generation.allow_non_adjacent_keys = gen_data[
                    "allow_non_adjacent_keys"
                ]

        # Parse corpus generation
        if "corpus_generation" in data:
            corpus_data = data["corpus_generation"]
            if "sample_size" in corpus_data:
                config.corpus_generation.sample_size = corpus_data["sample_size"]
            if "min_length" in corpus_data:
                config.corpus_generation.min_length = corpus_data["min_length"]
            if "max_length" in corpus_data:
                config.corpus_generation.max_length = corpus_data["max_length"]
            if "total_corpus_size" in corpus_data:
                config.corpus_generation.total_corpus_size = corpus_data[
                    "total_corpus_size"
                ]
            if "categories" in corpus_data:
                config.corpus_generation.categories = corpus_data["categories"]
            if "api_keys" in corpus_data:
                config.corpus_generation.api_keys = corpus_data["api_keys"]

        # Parse weights (we'll implement a simpler version for brevity)
        if "standalone_weights" in data:
            for key, value in data["standalone_weights"].items():
                metric_type = StandaloneMetricType[key]
                config.standalone_weights.weights[metric_type] = float(value)

        if "assignment_weights" in data:
            for key, value in data["assignment_weights"].items():
                metric_type = AssignmentMetricType[key]
                config.assignment_weights.weights[metric_type] = float(value)

        if "set_weights" in data:
            for key, value in data["set_weights"].items():
                metric_type = SetMetricType[key]
                config.set_weights.weights[metric_type] = float(value)

        # Parse token analysis config
        if "token_analysis" in data:
            token_data = data["token_analysis"]
            for key, value in token_data.items():
                if hasattr(config.token_analysis, key):
                    setattr(config.token_analysis, key, value)

        # Parse chord assignment config
        if "chord_assignment" in data:
            assign_data = data["chord_assignment"]
            for key, value in assign_data.items():
                if hasattr(config.chord_assignment, key):
                    setattr(config.chord_assignment, key, value)

        # Active settings
        if "active_layout" in data:
            config.active_layout = data["active_layout"]
        if "active_corpus" in data:
            config.active_corpus = data["active_corpus"]
        if "active_tokens" in data:
            config.active_tokens = data["active_tokens"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for YAML serialization"""
        # Create a dictionary with all configuration sections
        result = {
            "paths": {
                "key_layouts_dir": str(self.paths.key_layouts_dir),
                "corpuses_dir": str(self.paths.corpuses_dir),
                "tokens_dir": str(self.paths.tokens_dir),
                "chords_dir": str(self.paths.chords_dir),
                "debug_dir": str(self.paths.debug_dir),
                "results_dir": str(self.paths.results_dir),
                "cache_dir": str(self.paths.cache_dir),
                "default_layout_file": str(self.paths.default_layout_file),
                "default_corpus_file": str(self.paths.default_corpus_file),
                "default_tokens_file": str(self.paths.default_tokens_file),
            },
            "debug": {
                "enabled": self.debug.enabled,
                "log_level": self.debug.log_level.name,
                "log_file": str(self.debug.log_file) if self.debug.log_file else None,
                "print_cost_details": self.debug.print_cost_details,
                "save_intermediate_results": self.debug.save_intermediate_results,
            },
            "benchmark": {
                "enabled": self.benchmark.enabled,
                "track_individual_metrics": self.benchmark.track_individual_metrics,
                "visual_update_interval": self.benchmark.visual_update_interval,
            },
            "chord_generation": {
                "min_letter_count": self.chord_generation.min_letter_count,
                "max_letter_count": self.chord_generation.max_letter_count,
                "allow_non_adjacent_keys": self.chord_generation.allow_non_adjacent_keys,
            },
            "corpus_generation": {
                "sample_size": self.corpus_generation.sample_size,
                "min_length": self.corpus_generation.min_length,
                "max_length": self.corpus_generation.max_length,
                "total_corpus_size": self.corpus_generation.total_corpus_size,
                "categories": self.corpus_generation.categories,
                "api_keys": self.corpus_generation.api_keys,
            },
            "standalone_weights": {
                metric_type.name: weight
                for metric_type, weight in self.standalone_weights.weights.items()
            },
            "assignment_weights": {
                metric_type.name: weight
                for metric_type, weight in self.assignment_weights.weights.items()
            },
            "set_weights": {
                metric_type.name: weight
                for metric_type, weight in self.set_weights.weights.items()
            },
            "token_analysis": {
                "min_token_length": self.token_analysis.min_token_length,
                "max_token_length": self.token_analysis.max_token_length,
                "top_n_tokens": self.token_analysis.top_n_tokens,
                "include_characters": self.token_analysis.include_characters,
                "include_character_ngrams": self.token_analysis.include_character_ngrams,
                "include_words": self.token_analysis.include_words,
                "include_word_ngrams": self.token_analysis.include_word_ngrams,
                "use_parallel_processing": self.token_analysis.use_parallel_processing,
            },
            "chord_assignment": {
                "algorithm": self.chord_assignment.algorithm,
                "first_letter_unmatched_weight": self.chord_assignment.first_letter_unmatched_weight,
                "second_letter_unmatched_weight": self.chord_assignment.second_letter_unmatched_weight,
                "last_letter_unmatched_weight": self.chord_assignment.last_letter_unmatched_weight,
                "additional_letter_weight": self.chord_assignment.additional_letter_weight,
                "fallback_letter_weight": self.chord_assignment.fallback_letter_weight,
                "vertical_stretch_weight": self.chord_assignment.vertical_stretch_weight,
                "vertical_pinch_weight": self.chord_assignment.vertical_pinch_weight,
                "horizontal_stretch_weight": self.chord_assignment.horizontal_stretch_weight,
                "horizontal_pinch_weight": self.chord_assignment.horizontal_pinch_weight,
                "diagonal_stretch_weight": self.chord_assignment.diagonal_stretch_weight,
                "diagonal_pinch_weight": self.chord_assignment.diagonal_pinch_weight,
                "same_finger_double_weight": self.chord_assignment.same_finger_double_weight,
                "same_finger_triple_weight": self.chord_assignment.same_finger_triple_weight,
                "pinky_ring_stretch_weight": self.chord_assignment.pinky_ring_stretch_weight,
                "ring_middle_scissor_weight": self.chord_assignment.ring_middle_scissor_weight,
                "middle_index_stretch_weight": self.chord_assignment.middle_index_stretch_weight,
            },
            "active_layout": self.active_layout,
            "active_corpus": self.active_corpus,
            "active_tokens": self.active_tokens,
        }
        return result

    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        config_path = Path(config_path)

        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def setup_logging(self) -> None:
        """Configure logging based on debug settings"""
        if not self.debug.enabled:
            return

        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Ensure log directory exists
        if self.debug.log_file:
            log_dir = self.debug.log_file.parent
            os.makedirs(log_dir, exist_ok=True)

            logging.basicConfig(
                level=self.debug.log_level.value,
                format=log_format,
                handlers=[
                    logging.FileHandler(self.debug.log_file),
                    logging.StreamHandler(),
                ],
            )
        else:
            logging.basicConfig(level=self.debug.log_level.value, format=log_format)

    def validate(self) -> None:
        """Validate configuration values

        Raises:
            ValueError: If any configuration values are invalid
        """
        # Validate chord generation settings
        if self.chord_generation.min_letter_count < 1:
            raise ValueError("min_letter_count must be at least 1")
        if (
            self.chord_generation.max_letter_count
            < self.chord_generation.min_letter_count
        ):
            raise ValueError(
                "max_letter_count must be greater than or equal to min_letter_count"
            )

        # Validate token analysis settings
        if self.token_analysis.min_token_length < 1:
            raise ValueError("min_token_length must be at least 1")
        if self.token_analysis.max_token_length < self.token_analysis.min_token_length:
            raise ValueError(
                "max_token_length must be greater than or equal to min_token_length"
            )
        if self.token_analysis.top_n_tokens < 1:
            raise ValueError("top_n_tokens must be at least 1")

        # Validate corpus generation settings
        if self.corpus_generation.min_length < 1:
            raise ValueError("min_length must be at least 1")
        if self.corpus_generation.max_length < self.corpus_generation.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
        if self.corpus_generation.sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if self.corpus_generation.total_corpus_size < 1:
            raise ValueError("total_corpus_size must be at least 1")
        if not self.corpus_generation.categories:
            raise ValueError("categories must not be empty")
        if sum(self.corpus_generation.categories.values()) <= 0:
            raise ValueError("sum of category weights must be positive")

        # Validate paths
        if not self.paths.default_layout_file.exists():
            raise ValueError(
                f"Default layout file does not exist: {self.paths.default_layout_file}"
            )

        # Check active layout file exists
        layout_path = self.paths.get_layout_path(self.active_layout)
        if not layout_path.exists():
            raise ValueError(f"Active layout file does not exist: {layout_path}")
