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

def save_image(image: np.ndarray, image_path: Path) -> None:
    cv2.imwrite(str(image_path), image)

def load_np(filepath: Path, dtype: type = float) -> np.ndarray:
    if filepath.suffix == ".txt":
        return np.loadtxt(filepath, dtype=dtype)
    return np.load(filepath)

def save_np(array: np.ndarray, filepath: Path, format: str = ".txt") -> None:
    if format == ".txt":
        if array.dtype.type is np.str_:
            np.savetxt(str(filepath) + format, array, fmt='%s')
        else:
            np.savetxt(str(filepath) + format, array)
    else:
        np.save(filepath, array)
        