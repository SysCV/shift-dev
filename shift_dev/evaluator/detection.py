"""Detection evaluator."""

from __future__ import annotations

from .base import Evaluator

import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class DetectionEvaluator(Evaluator):

    METRICS = ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]

    def __init__(
        self, 
        num_classes: int = 5,
        iou_thresholds: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        bbox_format: str = "xywh",
        iou_type: str = "bbox",
    ) -> None:
        """Initialize the detection evaluator."""
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.bbox_format = bbox_format
        self.iou_type = iou_type
        assert self.bbox_format in ["xywh", "xyxy"], "Invalid bbox format."
        assert self.iou_type in ["bbox", "segm"], "Invalid iou type."
        super().__init__()

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self._predictions = []
        self._targets = []

    def process(self, prediction: dict[str, np.array], target: dict[str, np.array]) -> None:
        """Process a batch of data.

        Args:
            prediction (np.array): Prediction data dictionary.
            target (np.array): Target data dictionary.
        
        Note:
            The data dictionary should contain the following keys:
                - bbox: Bounding box array of shape (4,) in the format [x, y, w, h].
                - class_probs: Class logits array of shape (num_classes,).
                - score: Prediction score.
                - mask: Segmentation mask, in shape (H, W), with 0 for background and 1 for foreground.
        """
        # convert to COCO format
        if self.bbox_format == "xyxy":
            # convert to [x, y, w, h] format
            prediction["bbox"][:, 2] -= prediction["bbox"][:, 0]
            prediction["bbox"][:, 3] -= prediction["bbox"][:, 1]
            target["bbox"][:, 2] -= target["bbox"][:, 0]
            target["bbox"][:, 3] -= target["bbox"][:, 1]

        class_predicted = np.argmax(prediction["class_probs"], axis=-1)
        class_score = np.max(prediction["class_probs"], axis=-1)
        pred = {
            "image_id": len(self._predictions),
            "bbox": prediction["bbox"],
            "category_id": class_predicted,
            "score": class_score,
        }
        tgt = {
            "image_id": len(self._targets),
            "category_id": target["category_id"],
            "bbox": target["bbox"],
        }

        if "segmentation" in prediction:
            pred["segmentation"] = maskUtils.encode(
                np.array(prediction["mask"], order="F", dtype=np.uint8)
            )
            tgt["segmentation"] = maskUtils.encode(
                np.array(target["mask"], order="F", dtype=np.uint8)
            )
        self._predictions.append(pred)
        self._targets.append(tgt)

    def evaluate(self) -> dict[str, float]:
        """Evaluate all predictions according to given metric.

        Returns:
            dict[str, float]: Evaluation results.
        """
        coco_pred = COCO()
        coco_pred.dataset["images"] = [img for img in self._predictions]
        coco_pred.dataset["categories"] = [{"id": i} for i in range(self.num_classes)]
        coco_pred.createIndex()

        coco_gt = COCO()
        coco_gt.dataset["images"] = [img for img in self._targets]
        coco_gt.dataset["categories"] = [{"id": i} for i in range(self.num_classes)]
        coco_gt.createIndex()

        coco_eval = COCOeval(coco_gt, coco_pred, self.iou_type)
        coco_eval.params.iouThrs = self.iou_thresholds
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "mAP_s": coco_eval.stats[3],
            "mAP_m": coco_eval.stats[4],
            "mAP_l": coco_eval.stats[5],
        }
