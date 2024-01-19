from typing import Dict, List, Optional

import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch


def show_tranformed_image(train_loader: DataLoader) -> None:
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
            cv2.imshow('Transformed image', cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def visualize_sample(image: torch.Tensor, target: Dict[str, torch.Tensor], 
                     classes: List[str], plot: bool = True) -> Optional[np.ndarray]:
        image = image.permute(1, 2, 0).cpu().numpy()
        for box, label in zip(target['boxes'], target['labels']):
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 1
            )
            cv2.putText(
                image, classes[label], (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        if plot:
            cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def get_prediction(image: np.ndarray, prediction: Dict[str, torch.Tensor], 
                   classes: List[str], threshold: float) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score.item() < threshold:
                continue
        box = box.cpu().numpy()
        label = label.cpu().numpy()
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
        )
        cv2.putText(
            image, classes[label], (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    return image
