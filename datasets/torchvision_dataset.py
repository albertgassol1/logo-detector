from pathlib import Path
from typing import Dict, Tuple, List

import torch
import cv2
import numpy as np 
from torch.utils.data import Dataset, DataLoader

from utils.configs import DatasetConfig
from utils.io import load_json, read_txt_file, load_image, save_as_json
from utils.transforms import train_augmentation, validation_transform


class FlickrVisionDataset(Dataset):

    def __init__(self, config: DatasetConfig, classes_map: List[str], train: bool = True) -> None:

        self.config = config
        self.classes_map = classes_map
        self.train = train

        # Get data info
        self.images_info = load_json(config.train_file) if train else load_json(config.validation_file)
        self.images_names_list = list(self.images_info.keys())  
        
        self.augmentation = train_augmentation() if train else validation_transform()
    
    def __len__(self) -> int:
        return len(self.images_names_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        image_info = self.images_info[self.images_names_list[idx]]
        image = load_image(self.config.image_dir / self.images_names_list[idx])
        height, width, _ = image.shape
        image = cv2.resize(image, (self.config.width, self.config.height))
        image = image.astype(np.float32) / 255.0

        # Get resized bounding boxes
        boxes = torch.as_tensor([[bbox["x1"] * self.config.width / width, 
                                  bbox["y1"] * self.config.height / height, 
                                  bbox["x2"] * self.config.width / width, 
                                  bbox["y2"] * self.config.height / height] for bbox in image_info["bbox"]], dtype=torch.float32)


        # Prepare target
        target = dict()
        target["boxes"] = boxes
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["labels"] = torch.as_tensor([self.classes_map.index(object_class) for object_class in image_info["class"]], dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        # Data augmentation
        if self.config.augmentation:
            augmentation = self.augmentation(image=image, bboxes=target["boxes"], labels=target["labels"])
            image = augmentation["image"]
            target["boxes"] = torch.as_tensor(augmentation["bboxes"], dtype=torch.float32)

        return image, target
    
    def get_loader(self) -> DataLoader:
        return DataLoader(self, batch_size=self.config.batch_size, 
                          shuffle=self.train, num_workers=self.config.num_workers, 
                          collate_fn=self.collate_fn)
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        return tuple(zip(*batch))

    @staticmethod
    def split_dataset(train_percentage: float, annotation_file: Path,
                      subset: int = 6) -> Tuple[Path, Path, List[str]]:
        # Parse annotation file and get train and test subsets
        data = read_txt_file(annotation_file)
        all_data = dict()
        images_per_class = dict()
        for image_info in data:
            image_subset = int(image_info[2])
            if image_subset != subset:
                continue
            if (int(image_info[3]) >= int(image_info[5])) or (int(image_info[4]) >= int(image_info[6])):
                continue
            if image_info[0] not in all_data.keys():
                all_data[image_info[0]] = {"class": [image_info[1]],
                                           "bbox": [{"x1": int(image_info[3]),
                                                     "y1": int(image_info[4]),
                                                     "x2": int(image_info[5]),
                                                     "y2": int(image_info[6])}]}
                if image_info[1] not in images_per_class.keys():
                    images_per_class[image_info[1]] = [image_info[0]]
                else:
                    images_per_class[image_info[1]].append(image_info[0])
            else:
                all_data[image_info[0]]["bbox"].append({"x1": int(image_info[3]),
                                                        "y1": int(image_info[4]),
                                                        "x2": int(image_info[5]),
                                                        "y2": int(image_info[6])})
                all_data[image_info[0]]["class"].append(image_info[1])
        
        # Generate classes list
        classes = sorted(list(images_per_class.keys()))

        # Split train and validation
        train_data = dict()
        validation_data = dict()
        for image_names in images_per_class.values():
            n_train = int(len(image_names) * train_percentage)
            assert n_train > 0, "Train percentage is too low"
            assert n_train < len(image_names), "Train percentage is too high"
            train_names = np.random.choice(image_names, n_train, replace=False)
            val_names = [image_name for image_name in image_names if image_name not in train_names]
            train_data.update({image_name: all_data[image_name] for image_name in train_names})
            validation_data.update({image_name: all_data[image_name] for image_name in val_names})
        
        # Shuffle data
        train_keys = list(train_data.keys())
        np.random.shuffle(train_keys)
        train_data = {key: train_data[key] for key in train_keys}
        validation_keys = list(validation_data.keys())
        np.random.shuffle(validation_keys)
        validation_data = {key: validation_data[key] for key in validation_keys}

        # Write files to json
        save_as_json(annotation_file.parent / "train.json", train_data)
        save_as_json(annotation_file.parent / "validation.json", validation_data)

        return annotation_file.parent / "train.json", annotation_file.parent / "validation.json", classes
