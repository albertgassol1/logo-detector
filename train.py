import os
import argparse
from pathlib import Path
from typing import Optional

import torch

from datasets.fetch_dataset import fetch_dataset, DATASET_NAME
from datasets.torchvision_dataset import FlickrVisionDataset
from models import models_dict
from trainers.torchvision_trainer import VisionTrainer
from utils.configs import Config
from utils.config_parser import parse_config
from utils.visualization import show_tranformed_image, visualize_sample


def train(config: Config, dataset_path: Path, download: bool, 
          resume: Optional[Path], output_dir: Path) -> None:
    
    # Fetch dataset
    if download:
        config.dataset.image_dir, general_annotation = fetch_dataset(dataset_path)
    else:
        for file in (dataset_path / DATASET_NAME).iterdir():
            if file.is_dir():
                config.dataset.image_dir = file
            elif file.stem == "flickr_logos_27_dataset_training_set_annotation":
                general_annotation = file

    # Split dataset
    config.dataset.train_file, config.dataset.validation_file, classes_map = \
        FlickrVisionDataset.split_dataset(config.dataset.train_percentage, 
                                          general_annotation, config.dataset.subset)
    # Create datasets
    train_loader = FlickrVisionDataset(config.dataset, classes_map).get_loader()
    val_loader = FlickrVisionDataset(config.dataset, classes_map, train=False).get_loader()
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create model
    config.model.num_classes = len(classes_map)
    model = models_dict[config.model.type](config.model)
    if resume is not None:
        if resume.exists():
            model.load_state_dict(torch.load(resume, map_location=config.train.device))
    model.to(config.train.device)
    model.train()

    # Visualize augmentation
    if config.train.viz_augmentation:
        image, target = val_loader.dataset[0]
        visualize_sample(image, target, classes_map)
        show_tranformed_image(train_loader)

    # Create trainer
    trainer = VisionTrainer(model, config.train, train_loader, val_loader)
    trainer.train(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection model on Flickr Logos 27 dataset")
    parser.add_argument("--config", type=Path, required=True , help="Path to config file")
    parser.add_argument("--dataset-path", type=Path, default="data", help="Path to dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Path to output directory")
    args = parser.parse_args().__dict__
    
    script_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    args['dataset_path'] = script_folder / args['dataset_path']
    args['output_dir'] = script_folder / args['output_dir']

    config = parse_config(script_folder / args.pop('config'), args.pop('gpu'))
    
    train(**args, config=config)
