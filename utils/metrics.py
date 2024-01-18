from typing import Dict, List

import torch
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection import MeanAveragePrecision

from utils.configs import MetricsConfig


class Metrics:
    def __init__(self, config: MetricsConfig) -> None:
        self.config = config

        self.iou = IntersectionOverUnion(**config.params["IoU"])
        self.map = MeanAveragePrecision(**config.params["mAP"])
        self.average_metrics = {}
        self.iters = 0

    def reset(self) -> None:
        self.average_metrics = {}
        self.iters = 0

    def update(self, predictions: List[Dict[str, torch.Tensor]],
               targets: List[Dict[str, torch.Tensor]]) -> None:
        assert len(predictions) == len(targets), "Number of predictions and targets must be equal"
        for prediction, target in zip(predictions, targets):
            metrics = self._get_metrics(prediction, target)
            for metric, value in metrics.items():
                if metric not in self.average_metrics:
                    self.average_metrics[metric] = value
                else:
                    self.average_metrics[metric] += value
            self.iters += 1

    def compute(self) -> Dict:
        return {metric: value / self.iters for metric, value in self.average_metrics.items()}
    
    def compute_and_reset(self) -> Dict:
        metrics = self.compute()
        self.reset()
        return metrics

    def _get_metrics(self, prediction: Dict[str, torch.Tensor],
                     target: Dict[str, torch.Tensor]) -> Dict:
        
        metrics = {}
        if "IoU" in self.config.metrics:
            preds = [{
                "boxes": prediction["boxes"].cpu(),
                "labels": prediction["labels"].cpu()
            }]
            targets = [{
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu()
            }]
            metrics.update(self.iou(preds, targets))
        if "mAP" in self.config.metrics:
            preds = [{
                "boxes": prediction["boxes"].cpu(),
                "labels": prediction["labels"].cpu(),
                "scores": prediction["scores"].cpu()
            }]
            targets = [{
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu()
            }]
            metrics.update(self.map(preds, targets))

        metrics = {metric: value if not torch.isnan(value) else 0.0 for metric, value in metrics.items()}
        return metrics
