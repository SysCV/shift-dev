"""Optical flow estimation evaluator."""

from __future__ import annotations

from .base import Evaluator

import numpy as np


class OpticalFlowEvaluator(Evaluator):

    METRICS = ["epe"]

    def __init__(self, max_flow: float = 400.0) -> None:
        """Initialize the optical flow evaluator."""
        self.max_flow = max_flow
        super().__init__()

    def end_point_error(self, pred, target):
        """Compute the end point error.

        Args:
            pred (np.array): Prediction optical flow, in shape (H, W, 2).
            target (np.array): Target optical flow, in shape (H, W, 2).

        Returns:
            float: End point error.
        """
        mask = np.sum(np.abs(target), axis=2) < self.max_flow
        return np.mean(np.sqrt(np.sum((pred[mask] - target[mask]) ** 2, axis=1)))
    
    def process(self, prediction: np.array, target: np.array) -> None:
        """Process a batch of data.

        Args:
            prediction (np.array): Prediction optical flow, in shape (H, W, 2).
            target (np.array): Target optical flow, in shape (H, W, 2).
        """
        epe = self.end_point_error(prediction, target)
        self.metrics.update({"epe": epe})
        