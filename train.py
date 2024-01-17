import os
import argparse
from copy import deepcopy
from pathlib import Path

from datasets.fetch_dataset import fetch_dataset
from datasets.torchvision_dataset import FlickrVisionDataset
from models import models_dict
from utils.configs import Config
from utils.config_parser import parse_config

def train(config: Config, dataset_path: Path, download: bool, 
          resume: str, output_dir: Path) -> None:
    
    # Fetch dataset
    if download:
        config.dataset.image_dir, general_annotation = fetch_dataset(dataset_path)
    else:
        for file in dataset_path.iterdir():
            if file.is_dir():
                config.dataset.image_dir = file
            elif file.stem == "flickr_logos_27_dataset_training_set_annotation":
                general_annotation = file

    # Split dataset
    train_annotation, val_annotation = FlickrVisionDataset.split_dataset(config.dataset.train_percentage, 
                                                                         general_annotation)
    # Get classes
    classes = FlickrVisionDataset.get_classes(general_annotation)
    
    # Create datasets
    train_dataset = FlickrVisionDataset(config.dataset, classes)
    val_config = deepcopy(config.dataset)
    val_config.augmentation = False
    val_dataset = FlickrVisionDataset(val_config, classes, train=False)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # Create model
    model = models_dict[config.model.type](config)
    model.to(config.train.device)
    model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection model on Flickr Logos 27 dataset")
    parser.add_argument("--config", type=Path, required=True , help="Path to config file")
    parser.add_argument("--dataset-path", type=Path, default="data", help="Path to dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU to use")
    # parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    # parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    # parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    # parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    # parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    # parser.add_argument("--print-freq", type=int, default=10, help="Print frequency")
    # parser.add_argument("--save-freq", type=int, default=10, help="Save frequency")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Path to output directory")
    args = parser.parse_args().__dict__
    
    script_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    args['dataset_path'] = script_folder / args['dataset_path']
    args['output_dir'] = script_folder / args['output_dir']

    config = parse_config(script_folder / args.pop('config'), args.pop('gpu'))
    
    train(**args, config=config)