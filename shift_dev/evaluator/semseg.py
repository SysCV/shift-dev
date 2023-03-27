"""Semantic segmentation evaluation."""

from __future__ import annotations

from .base import Evaluator

import numpy as np


class SemanticSegmentationEvaluator(Evaluator):

    METRICS = ["mIoU", "mAcc", "fwIoU", "fwAcc"]

    def __init__(self, num_classes: int = 21, class_to_ignore: int = 255) -> None:
        """Initialize the semantic segmentation evaluator."""
        self.num_classes = num_classes
        super().__init__()

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self._confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def calc_confusion_matrix(self, prediction: np.array, target: np.array) -> np.array:
        """Calculate the confusion matrix.

        Args:
            prediction (np.array): Prediction semantic segmentation map, in shape (H, W).
            target (np.array): Target semantic segmentation map, in shape (H, W).

        Returns:
            np.array: Confusion matrix.
        """
        mask = (target >= 0) & (target < self.num_classes) & (target != self.class_to_ignore)
        return np.bincount(
            self.num_classes * target[mask].astype(np.uint8) + prediction[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def process(self, prediction: np.array, target: np.array) -> None:
        """Process a batch of data.

        Args:
            prediction (np.array): Prediction semantic segmentation map.
            target (np.array): Target semantic segmentation map.
        """
        self._confusion_matrix += self.calc_confusion_matrix(prediction, target)

    def evaluate(self) -> dict[str, float]:
        """Evaluate all predictions according to given metric.

        Returns:
            dict[str, float]: Evaluation results.
        """
        confusion_matrix = self._confusion_matrix
        iou = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
        )
        mean_iou = np.nanmean(iou)
        mean_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        mean_fw_iou = np.nansum(iou * np.sum(confusion_matrix, axis=1)) / np.sum(confusion_matrix)
        mean_fw_acc = np.nansum(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)) / self.num_classes

        return {
            "mIoU": mean_iou,
            "mAcc": mean_acc,
            "fwIoU": mean_fw_iou,
            "fwAcc": mean_fw_acc,
        }