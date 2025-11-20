"""
Utilities for extracting stable voltage plateaus from CSV files and
plotting the normal distribution of each plateau.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class StableSegment:
    """Represents a stable plateau in the voltage trace."""

    start_index: int
    end_index: int
    mean_voltage: float
    std_voltage: float

    def as_slice(self) -> slice:
        return slice(self.start_index, self.end_index + 1)


@dataclass
class StabilityConfig:
    """Configuration for plateau detection and visualization."""

    window_size: int = 50
    std_threshold: float = 0.5
    min_length: int = 50
    voltage_column: str = "voltage"
    time_column: Optional[str] = None


class StableSegmentProcessor:
    """Processes voltage traces and extracts stable plateaus."""

    def __init__(self, config: StabilityConfig | None = None):
        self.config = config or StabilityConfig()

    def load_voltage_series(self, csv_path: Path) -> pd.Series:
        df = pd.read_csv(csv_path)
        if self.config.voltage_column not in df.columns:
            raise ValueError(
                f"CSV file must contain a '{self.config.voltage_column}' column."
            )
        return df[self.config.voltage_column]

    def detect_stable_segments(self, series: pd.Series) -> List[StableSegment]:
        if len(series) < self.config.window_size:
            return []

        rolling_std = series.rolling(self.config.window_size, center=True).std()
        is_stable = rolling_std < self.config.std_threshold

        segments: List[StableSegment] = []
        current_start: Optional[int] = None

        for idx, stable in enumerate(is_stable):
            if stable and current_start is None:
                current_start = idx
            elif not stable and current_start is not None:
                self._maybe_add_segment(series, segments, current_start, idx - 1)
                current_start = None

        if current_start is not None:
            self._maybe_add_segment(series, segments, current_start, len(series) - 1)

        return segments

    def _maybe_add_segment(
        self,
        series: pd.Series,
        segments: List[StableSegment],
        start: int,
        end: int,
    ) -> None:
        if end - start + 1 < self.config.min_length:
            return
        segment_values = series.iloc[start : end + 1]
        mean_voltage = float(segment_values.mean())
        std_voltage = float(segment_values.std(ddof=0))
        segments.append(StableSegment(start, end, mean_voltage, std_voltage))

    def summarize_segments(self, series: pd.Series) -> List[StableSegment]:
        segments = self.detect_stable_segments(series)
        return segments

    def plot_segment_distribution(
        self,
        segment: StableSegment,
        series: pd.Series,
        output_path: Path,
        bins: int = 30,
    ) -> None:
        values = series.iloc[segment.as_slice()]
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(values, bins=bins, density=True, alpha=0.6, label="数据直方图")

        x_vals = np.linspace(values.min(), values.max(), 200)
        pdf_vals = norm.pdf(x_vals, loc=segment.mean_voltage, scale=segment.std_voltage)
        ax.plot(x_vals, pdf_vals, "r-", label="正态分布拟合")

        ax.axvline(segment.mean_voltage, color="k", linestyle="--", label="平均值")
        ax.set_xlabel("电压值")
        ax.set_ylabel("概率密度")
        ax.set_title(
            f"稳定区间 {segment.start_index}-{segment.end_index}\n"
            f"均值={segment.mean_voltage:.3f}, 标准差={segment.std_voltage:.3f}"
        )
        ax.legend()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def process_file(
    csv_path: Path,
    output_dir: Path,
    config: StabilityConfig | None = None,
) -> Tuple[List[StableSegment], pd.Series]:
    """
    Process a single CSV file and produce plots for each stable segment.

    Parameters
    ----------
    csv_path: Path
        Input CSV file containing at least a voltage column.
    output_dir: Path
        Directory where per-segment plots will be saved.
    config: StabilityConfig | None
        Optional configuration overrides.

    Returns
    -------
    Tuple[List[StableSegment], pd.Series]
        Detected segments and the original voltage series for further use.
    """

    processor = StableSegmentProcessor(config)
    series = processor.load_voltage_series(csv_path)
    segments = processor.summarize_segments(series)

    for idx, segment in enumerate(segments, start=1):
        plot_path = output_dir / f"{csv_path.stem}_segment{idx}.png"
        processor.plot_segment_distribution(segment, series, plot_path)

    return segments, series


def demo():
    """Run a small demo using a synthetic stepped signal."""
    rng = np.random.default_rng(42)
    steps = [
        (0, 100, 0.2),
        (3.3, 120, 0.05),
        (1.8, 90, 0.1),
        (5.0, 110, 0.07),
    ]

    values = []
    for voltage, length, noise in steps:
        values.append(voltage + rng.normal(0, noise, size=length))
    series = pd.Series(np.concatenate(values))

    processor = StableSegmentProcessor()
    segments = processor.summarize_segments(series)

    output_dir = Path("demo_plots")
    output_dir.mkdir(exist_ok=True)
    for i, segment in enumerate(segments, start=1):
        processor.plot_segment_distribution(
            segment, series, output_dir / f"demo_segment_{i}.png"
        )

    for segment in segments:
        print(
            f"Segment {segment.start_index}-{segment.end_index}: "
            f"mean={segment.mean_voltage:.3f}, std={segment.std_voltage:.3f}"
        )


if __name__ == "__main__":
    demo()
