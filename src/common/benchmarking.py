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


@dataclass
class PhaseMetrics:
    """Metrics collected for each processing phase"""

    elapsed_time: float
    memory_delta: float
    items_processed: int


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
        self._last_display_update = 0

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

        if self.config.track_generation_phases:
            self._update_display()

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

        if (
            self.config.track_generation_phases
            and current_time - self._last_display_update >= self.config.sample_interval
        ):
            self._update_display()
            self._last_display_update = current_time

    def end_phase(self) -> None:
        """Complete tracking for the current phase"""
        if not self.enabled or not self._current_phase:
            return

        self._current_phase = None

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
        # Clear screen
        if sys.platform.startswith("win"):
            os.system("cls")
        else:
            os.system("clear")

        print(f"\nBenchmark Status - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)

        headers = ["Phase", "Items", "Time (s)"]
        if self.config.include_memory_stats:
            headers.append("Memory Δ (MB)")

        rows: List[List[Any]] = []
        for phase, metrics in self._phase_metrics.items():
            row = [
                f"► {phase}" if phase == self._current_phase else phase,
                f"{metrics.items_processed:,}",
                f"{metrics.elapsed_time:.2f}",
            ]

            if self.config.include_memory_stats:
                row.append(f"{metrics.memory_delta:.2f}")

            rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()
        sys.stdout.flush()

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
