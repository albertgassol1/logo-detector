from pathlib import Path
from typing import Dict, Tuple, List

import torch
import cv2
import numpy as np 
from torch.utils.data import Dataset

from utils.configs import DatasetConfig
from utils.io import parse_annotation_file, write_annotation_file, load_image
from utils.transforms import train_augmentation, validation_transform

class FlickrVisionDataset(Dataset):

    def __init__(self, config: DatasetConfig, classes_map: List[str], train: bool = True) -> None:

        self.config = config
        self.classes_map = classes_map
        # Get train images
        self.images = list()
        # TODO: Change accordingly
        images_info = parse_annotation_file(self.config.annotation_file)[0]
        self.image_paths = list()
        self.classes = list()
        self.bboxes = list()
        for image_info in images_info:
            self.image_paths.append(self.config.image_dir / image_info["image_name"])
            self.classes.append(image_info["class"])
            self.bboxes.append([image_info["x1"], image_info["y1"], image_info["x2"], image_info["y2"]])
        
        self.augmentation = train_augmentation() if train else validation_transform()
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = load_image(self.image_paths[idx])
        height, width, _ = image.shape
        image = cv2.resize(image, (self.config.width, self.config.height))
        image = image.astype(np.float32) / 255.0
        class_idx = self.classes_map.index(self.classes[idx])
        bbox = self.bboxes[idx]

        # Prepare target
        target = dict()
        # TODO: Change to add multiple bboxes per image
        # Resizing bounding box coordinates
        boxes = torch.as_tensor([bbox[0] * self.config.width / width, 
                 bbox[1] * self.config.height / height, 
                 bbox[2] * self.config.width / width, 
                 bbox[3] * self.config.height / height], dtype=torch.float32)
        target["boxes"] = boxes
        target["area"] = torch.as_tensor([(boxes[3] - boxes[1]) * (boxes[2] - boxes[0])], dtype=torch.float32)
        target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)
        target["labels"] = torch.as_tensor([class_idx], dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        # Data augmentation
        if self.config.augmentation:
            augmentation = self.augmentation(image=image, bboxes=target["boxes"], labels=target["labels"])
            image = augmentation["image"]
            target["boxes"] = torch.as_tensor(augmentation["bboxes"], dtype=torch.float32)

        return image, target
        
    # TODO: Parse file here, write two files with different format. Json would be nice. Also compute classes here.
    @staticmethod
    def split_dataset(train_percentage: float, annotation_file: Path) -> Tuple[Path, Path]:
        # Parse annotation file and get train and test subsets
        data = parse_annotation_file(annotation_file)
        num_subsets_train = int(len(data.keys()) * train_percentage)
        assert num_subsets_train > 0, "Train percentage too low"
        assert num_subsets_train < len(data.keys()), "Train percentage too high"
        train_subsets_keys = np.random.choice(list(data.keys()), num_subsets_train, replace=False)
        train_subsets = {0: list(set().union(*[set(data[subset]) for subset in train_subsets_keys]))}
        test_subsets = {0: list(set().union(*[data[subset] for subset in data.keys() if subset not in train_subsets]))}
        # Write train and test subsets to file
        write_annotation_file(annotation_file.parent / "train.txt", train_subsets)
        write_annotation_file(annotation_file.parent / "test.txt", test_subsets)
        return annotation_file.parent / "train.txt", annotation_file.parent / "test.txt"
    
    @staticmethod
    def get_classes(annotation_file: Path) -> List[str]:
        data = parse_annotation_file(annotation_file)
        classes = set()
        for subset in data.keys():
            for image_info in data[subset]:
                classes.add(image_info["class"])
        return sorted(list(classes))
        