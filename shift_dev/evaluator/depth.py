"""Depth estimation evaluator."""

from __future__ import annotations

from .base import Evaluator

import numpy as np


class DepthEvaluator(Evaluator):

    METRICS = ["mae", "silog"]

    def __init__(self, min_depth: float = 0.5, max_depth: float = 80.0) -> None:
        """Initialize the depth evaluator."""
        self.min_depth = min_depth
        self.max_depth = max_depth
        super().__init__()

    def mean_absolute_error(self, pred, target):
        """Compute the mean absolute error.

        Args:
            pred (np.array): Prediction depth map, in shape (H, W).
            target (np.array): Target depth map, in shape (H, W).

        Returns:
            float: Mean absolute error.
        """
        mask = (target > self.min_depth) & (target < self.max_depth)
        return np.mean(np.abs(pred[mask] - target[mask]))
    
    def silog(self, pred, target, eps=1e-6):
        """Compute the scale-invariant log error of KITTI.

        Args:
            pred (np.array): Prediction depth map, in shape (H, W).
            target (np.array): Target depth map, in shape (H, W).
            eps (float, optional): Epsilon. Defaults to 1e-6.

        Returns:
            float: Silog error.
        """
        mask = (target > self.min_depth) & (target < self.max_depth)
        log_diff = np.log(target[mask] + eps) - np.log(pred[mask] + eps)
        return np.sqrt(np.mean(log_diff ** 2))
    
    def process(self, prediction: np.array, target: np.array) -> None:
        """Process a batch of data.

        Args:
            prediction (np.array): Prediction depth map.
            target (np.array): Target depth map.
        """
        mae = self.mean_absolute_error(prediction, target)
        silog = self.silog(prediction, target)
        self.metrics.update({"mae": mae, "silog": silog})

    def evaluate(self) -> dict[str, float]:
        """Evaluate all predictions according to given metric.

        Returns:
            dict[str, float]: Evaluation results.
        """
        return {metric: np.nanmean(self.metrics[metric]) for metric in self.metrics}