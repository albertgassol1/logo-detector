from pathlib import Path
import json
from typing import Dict, List

import cv2
import numpy as np


def read_txt_file(filepath: Path, separator: str = " ") -> List[str]:
    with filepath.open() as f:
        lines = f.readlines()
    return [line.split(separator) for line in lines]

def load_json(filepath: Path) -> Dict:
    with filepath.open() as f:
        data = json.load(f)
    return data

def save_as_json(filepath: Path, data: Dict) -> None:
    with filepath.open("w") as f:
        json.dump(data, f, indent=4)


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image