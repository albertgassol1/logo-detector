from pathlib import Path
import yaml

from utils.configs import *

def parse_config(config_path: Path, gpu: str) -> Config:
    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config = DatasetConfig(**config["dataset"])
    
    train_config = TrainConfig(**config["train"], gpu=gpu)
    model_config = ModelConfig(**config["model"])
    metrics_config = MetricsConfig(**config["metrics"])
    
    return Config(dataset=dataset_config, train=train_config,
                  model=model_config, metrics=metrics_config)
