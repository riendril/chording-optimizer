"""Benchmarking utilities for performance tracking"""

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import psutil
from tabulate import tabulate

from src.common.config import BenchmarkOptions


class BenchmarkPhase(Enum):
    """Standard processing phases for benchmarking"""

    INITIALIZATION = auto()
    CHORD_GENERATION = auto()
    WORD_ASSIGNMENT = auto()
    SET_IMPROVEMENT = auto()
    FINALIZATION = auto()


@dataclass
class PhaseMetrics:
    """Metrics collected for each processing phase"""

    elapsed_time: float
    memory_delta: float
    items_processed: int


@dataclass
class MetricStats:
    """Statistics for individual metric calculations"""

    total_time: float
    call_count: int


class Benchmark:
    """Core benchmarking functionality"""

    def __init__(self, config: BenchmarkOptions):
        self.config = config
        self.enabled = config.enabled
        self._start_time = time.time()
        self._start_memory = self._get_memory_usage()
        self._phase_metrics: Dict[BenchmarkPhase, PhaseMetrics] = {}
        self._current_phase: Optional[BenchmarkPhase] = None
        self._phase_start_times: Dict[BenchmarkPhase, float] = {}
        self._phase_start_memory: Dict[BenchmarkPhase, float] = {}
        self._last_items_count = 0  # Track items count for update interval
        self._metric_stats: Dict[str, MetricStats] = {}
        self._current_metric: Optional[str] = None
        self._metric_start_time: Optional[float] = None
        self._metric_calculations_since_update = 0

    def start_phase(self, phase: BenchmarkPhase) -> None:
        """Begin tracking a new processing phase"""
        if not self.enabled:
            return

        self._current_phase = phase
        self._phase_start_times[phase] = time.time()
        self._phase_start_memory[phase] = self._get_memory_usage()
        self._phase_metrics[phase] = PhaseMetrics(
            elapsed_time=0, memory_delta=0, items_processed=0
        )

        self._update_display()  # Always show display if enabled

    def update_phase(self, items_processed: int) -> None:
        """Update metrics for the current phase"""
        if not self.enabled or not self._current_phase:
            return

        phase = self._current_phase
        current_time = time.time()
        current_memory = self._get_memory_usage()

        self._phase_metrics[phase] = PhaseMetrics(
            elapsed_time=current_time - self._phase_start_times[phase],
            memory_delta=current_memory - self._phase_start_memory[phase],
            items_processed=items_processed,
        )

        # Update display if interval is reached
        if (
            items_processed - self._last_items_count
            >= self.config.visual_update_interval
        ):
            self._update_display()
            self._last_items_count = items_processed

    def end_phase(self) -> None:
        """Complete tracking for the current phase"""
        if not self.enabled or not self._current_phase:
            return

        self._current_phase = None

    def start_metric_calculation(self, metric_name: str) -> None:
        """Start timing an individual metric calculation"""
        if not (self.enabled and self.config.track_individual_metrics):
            return

        self._current_metric = metric_name
        self._metric_start_time = time.time()

        # Initialize stats if this is a new metric
        if metric_name not in self._metric_stats:
            self._metric_stats[metric_name] = MetricStats(total_time=0.0, call_count=0)

    def end_metric_calculation(self) -> None:
        """End timing the current metric calculation"""
        if (
            not (self.enabled and self.config.track_individual_metrics)
            or not self._current_metric
        ):
            return

        elapsed = time.time() - self._metric_start_time
        stats = self._metric_stats[self._current_metric]
        stats.total_time += elapsed
        stats.call_count += 1

        self._metric_calculations_since_update += 1

        # Update display if interval is reached
        if self._metric_calculations_since_update >= self.config.visual_update_interval:
            self._update_display()
            self._metric_calculations_since_update = 0

        self._current_metric = None
        self._metric_start_time = None

    def get_results(self) -> Dict[str, Any]:
        """Retrieve complete benchmark results"""
        if not self.enabled:
            return {}

        total_time = time.time() - self._start_time
        total_memory = self._get_memory_usage() - self._start_memory

        results = {
            "total_execution_time": total_time,
            "total_memory_change": total_memory,
            "phases": {
                name: {
                    "time_seconds": metrics.elapsed_time,
                    "memory_mb": metrics.memory_delta,
                    "processed_items": metrics.items_processed,
                }
                for name, metrics in self._phase_metrics.items()
            },
        }

        return results

    def _update_display(self) -> None:
        """Update the console display with current metrics"""
        if not self.enabled:
            return

        # Clear screen
        if sys.platform.startswith("win"):
            os.system("cls")
        else:
            os.system("clear")

        print(f"\nBenchmark Status - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)

        # Phase progress
        if self._phase_metrics:
            print("\nPhase Progress:")
            headers = ["Phase", "Items", "Time (s)", "Memory Δ (MB)"]
            rows = []
            for phase, metrics in self._phase_metrics.items():
                row = [
                    f"► {phase}" if phase == self._current_phase else phase,
                    f"{metrics.items_processed:,}",
                    f"{metrics.elapsed_time:.2f}",
                    f"{metrics.memory_delta:.2f}",
                ]
                rows.append(row)
            print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Individual metric stats
        if self.config.track_individual_metrics and self._metric_stats:
            print("\nMetric Performance:")
            headers = ["Metric", "Calls", "Total Time (s)", "Avg Time (ms)"]
            rows = []
            for name, stats in self._metric_stats.items():
                avg_time = (
                    (stats.total_time / stats.call_count) * 1000
                    if stats.call_count > 0
                    else 0
                )
                rows.append(
                    [
                        name,
                        f"{stats.call_count:,}",
                        f"{stats.total_time:.2f}",
                        f"{avg_time:.3f}",
                    ]
                )
            print(tabulate(rows, headers=headers, tablefmt="grid"))

        print()
        sys.stdout.flush()

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
