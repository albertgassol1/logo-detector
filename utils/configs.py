from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import GPUtil

@dataclass
class DatasetConfig:
    train_percentage: float
    test_percentage: float
    val_percentage: float
    batch_size: int
    num_workers: int
    augmentation: bool
    width: int
    height: int
    subset: int
    annotation_file: Path = Path("")
    train_file: Path = Path("")
    validation_file: Path = Path("")
    image_dir: Path = Path("")

    def __post_init__(self) -> None:
        assert (self.train_percentage + self.test_percentage + self.val_percentage) == 1.0, "Split percentages must sum 1"

@dataclass
class TrainConfig:
    optimizer: str
    scheduler: str
    scheduler_params: Dict
    epochs: int
    lr: float
    weight_decay: float
    momentum: float
    log_freq: int
    save_freq: int
    gpu: str 
    viz_augmentation: bool
    nesterov: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        if self.gpu == "auto":
            device_ids = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, 
                                             includeNan=False, excludeID=[], excludeUUID=[])
            if len(device_ids) > 0 and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device_ids[0]}")
                return
        self.device = torch.device("cpu")
       
@dataclass
class ModelConfig:
    type: str
    name: str
    freeze: bool
    num_classes: int = 0

@dataclass
class MetricsConfig:
    metrics: List = field(default_factory=lambda: [])
    params: Dict = field(default_factory=lambda: {})

    def __post_init__(self):
        if "IoU" not in self.params:
            self.params["IoU"] = {}
        if "mAP" not in self.params:
            self.params["mAP"] = {}
            
@dataclass
class Config:
    dataset: DatasetConfig
    train: TrainConfig
    model: ModelConfig
    metrics: MetricsConfig
