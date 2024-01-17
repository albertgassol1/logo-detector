from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

def parse_annotation_file(filepath: Path) -> Dict[int, List[Dict[str, str]]]:
    with filepath.open() as f:
        lines = f.readlines()
    ret = dict()
    for line in lines:
        data = line.split(" ")
        subset = int(data[2])
        image_info = {"image_name": data[0],
                      "class": data[1],
                      "x1": data[3],
                      "y1": data[4],
                      "x2": data[5],
                      "y2": data[6]}
        if subset in ret.keys():
            ret[subset].append(image_info)
        else:
            ret[subset] = [image_info]
    return ret

def write_annotation_file(filepath: Path, data: Dict[int, List[Dict[str, str]]]) -> None:
    with filepath.open("w") as f:
        for subset in data.keys():
            for image_info in data[subset]:
                f.write(f"{image_info['image_name']} {image_info['class']} {subset} {image_info['x1']} {image_info['y1']} {image_info['x2']} {image_info['y2']}\n")


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image