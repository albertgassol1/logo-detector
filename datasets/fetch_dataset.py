import os
from pathlib import Path
import requests
import tarfile
from io import BytesIO

DATASET_URL = "http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz"
DATASET_NAME = DATASET_URL.split("/")[-1].split(".")[0] 


def fetch_dataset(dataset_path: Path) -> None:
    dataset_path.mkdir(exist_ok=True, parents=True)

    # Download dataset
    print("Downloading dataset...")
    r = requests.get(DATASET_URL)
    tar_ref = tarfile.open(fileobj=BytesIO(r.content))
    tar_ref.extractall(dataset_path)
    tar_ref.close()

    images_zip = dataset_path / DATASET_NAME / "flickr_logos_27_dataset_images.tar.gz"
    # Extract images
    print("Extracting images...")
    tar_ref = tarfile.open(images_zip)
    tar_ref.extractall(dataset_path / DATASET_NAME)
    tar_ref.close()
    images_zip.unlink()

    return dataset_path / DATASET_NAME / "flickr_logos_27_dataset_images", \
           dataset_path / DATASET_NAME / "flickr_logos_27_dataset_training_set_annotation.txt"
