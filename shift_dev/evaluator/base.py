"""SHIFT base evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np


class Evaluator:
    """Abstract evaluator class."""

    METRICS: list[str] = []

    def __init__(self) -> None:
        """Initialize evaluator."""
        self.reset()

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self.metrics = {metric: [] for metric in self.METRICS}

    def process(self, *args: Any) -> None:  # type: ignore
        """Process a batch of data."""
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        """Evaluate all predictions according to given metric.

        Returns:
            dict[str, float]: Evaluation results.
        """
        return {metric: np.nanmean(values) for metric, values in self.metrics.items()}
