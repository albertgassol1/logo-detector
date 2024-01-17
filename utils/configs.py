from dataclasses import dataclass
from pathlib import Path
import torch
import GPUtil

@dataclass
class DatasetConfig:
    train_percentage: float
    augmentation: bool
    width: int
    height: int
    annotation_file: Path = Path("")
    image_dir: Path = Path("")

@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    log_freq: int
    save_freq: int
    gpu: str 
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
class Config:
    dataset: DatasetConfig
    train: TrainConfig
    model: ModelConfig
