from pathlib import Path
from typing import Dict, List

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
