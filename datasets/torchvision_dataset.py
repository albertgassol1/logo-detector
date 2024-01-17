import os
from pathlib import Path
from typing import Tuple

import torch
import cv2
import numpy as np 
from torch.utils.data import Dataset

from utils.io import read_txt_file

class FlickrVisionDataset(Dataset):

    def __init__(self, config) -> None:

        self.config = config
        
        # Get train images

    @staticmethod
    def split_dataset(train_percentage: float, annotation_file: Path) -> Tuple[Path, Path]:
        
        data = read_txt_file(annotation_file)
        
