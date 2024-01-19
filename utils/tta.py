from typing import Dict, List

import numpy as np
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
import imutils


def perform_tta(image: np.ndarray) -> List[torch.Tensor]:

    augmented_images = [image]
    for i in range(3):
        augmented_images.append(imutils.rotate(augmented_images[i].copy(), angle=90))
    return augmented_images

def rotate_boxes(boxes: torch.Tensor, angle: float, image_width: int, image_height: int) -> torch.Tensor:
    # Rotate bounding boxes
    cx, cy = image_width / 2, image_height / 2
    rad_angle = torch.deg2rad(torch.tensor(angle, dtype=torch.float32))
    cos_theta = torch.cos(rad_angle)
    sin_theta = torch.sin(rad_angle)

    # Shift coordinates to the image center
    boxes_centered = boxes - torch.tensor([cx, cy, cx, cy], dtype=torch.float32)
    
    # Rotate bounding boxes
    rotated_boxes_centered = torch.stack([
        cos_theta * boxes_centered[:, 0] - sin_theta * boxes_centered[:, 1],
        sin_theta * boxes_centered[:, 0] + cos_theta * boxes_centered[:, 1],
        cos_theta * boxes_centered[:, 2] - sin_theta * boxes_centered[:, 3],
        sin_theta * boxes_centered[:, 2] + cos_theta * boxes_centered[:, 3],
    ], dim=1)

    # Shift coordinates back to the original position
    rotated_boxes = rotated_boxes_centered + torch.tensor([cx, cy, cx, cy], dtype=torch.float32)

    return rotated_boxes

@torch.inference_mode()
def predict_with_tta(model: torch.nn.Module, image: np.ndarray) -> Dict:
    image = image.copy().astype(np.float32) / 255.0
    augmented_images = perform_tta(image)

    all_predictions = []
    for augmented_image in augmented_images:
        prediction = model([augmented_image])
        all_predictions.append(prediction)

    # Aggregate predictions (e.g., average)
    return average_predictions(all_predictions, image.shape[0], image.shape[1])

def average_predictions(predictions: List[Dict[str, torch.Tensor]], image_width: int, image_height: int) -> Dict:
    # Implement your aggregation strategy here
    # For example, you could average the bounding box coordinates
    averaged_boxes = torch.mean(torch.stack([rotate_boxes(pred[0]['boxes'], -30, image_width, image_height) for pred in predictions]), dim=0)
    averaged_scores = torch.mean(torch.stack([pred[0]['scores'] for pred in predictions]), dim=0)

    return [{'boxes': averaged_boxes, 'scores': averaged_scores}]
