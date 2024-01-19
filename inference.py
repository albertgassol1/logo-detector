import argparse
import os
from pathlib import Path

import torch

from models import models_dict
from utils.configs import Config
from utils.config_parser import parse_config
from utils.io import load_json, load_image, save_image, save_as_json
from utils.transforms import cv_image_to_tensor
from utils.visualization import get_prediction, visualize_sample
from utils.metrics import Metrics


@torch.inference_mode()
def inference(test_folder: Path, output_path: Path, config: Config,
              model_checkpoint: Path, eval_metrics: bool = True,
              tta: bool = True, detection_th: float = 0.7) -> None:
    
    if not model_checkpoint.exists():
        raise FileNotFoundError(f"{model_checkpoint} does not exist")
    checkpoint = torch.load(model_checkpoint, map_location=config.train.device)
    classes = checkpoint["classes"]
    config.model.num_classes = len(classes)
    model = models_dict[config.model.type](config.model)
    model.to(config.train.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_path.mkdir(parents=True, exist_ok=True)

    if eval_metrics and (test_folder / "test.json").exists():
        test_annotations = load_json(test_folder / "test.json")
    metrics = Metrics(config.metrics)

    for image_path in (test_folder / "images").iterdir():
        image = load_image(image_path)
        image_t = cv_image_to_tensor(image).to(config.train.device)

        # TODO: Implement TTA. Call it here.
        prediction = model(image_t.unsqueeze(0))[0]
        
        image_prediction = get_prediction(image, prediction, classes, detection_th)
        save_image(image_prediction, output_path / f"{image_path.stem}_pred{image_path.suffix}")


        if eval_metrics:
            image_info = test_annotations[image_path.name]
            targets = {"boxes": torch.as_tensor([[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]] 
                                                 for bbox in image_info["bbox"]], dtype=torch.float32),
                       "labels": torch.as_tensor([classes.index(object_class) for object_class in image_info["class"]], dtype=torch.int64)}
            metrics.update([prediction], [targets])
            gt = visualize_sample((image_t * 255).to(torch.uint8), targets, classes, False)
            save_image(gt, output_path / f"{image_path.stem}_gt{image_path.suffix}")

    if eval_metrics:
        mean_metrics = metrics.compute_and_reset()
        save_as_json(output_path / "metrics.json", mean_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection model on Flickr Logos 27 dataset")
    parser.add_argument("--config", type=Path, required=True , help="Path to config file")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_folder", type=Path, required=True , help="Path to folder where test images are")
    parser.add_argument("--output_path", type=str, default="inference_output", help="Path to output directory")
    parser.add_argument("--eval_metrics", action="store_true", help="Eval metrics using GT")
    parser.add_argument("--detection_th", type=float, default=0.7, help="Detection threshold")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU to use")
    args = parser.parse_args().__dict__

    script_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    args['model_checkpoint'] = script_folder / args['model_checkpoint']
    args['output_path'] = script_folder / args['output_path']
    args['test_folder'] = script_folder / args['test_folder']

    config = parse_config(script_folder / args.pop('config'), args.pop('gpu'))

    inference(**args, config=config)
            