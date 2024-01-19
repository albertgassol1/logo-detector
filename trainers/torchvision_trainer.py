from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import wandb

from models import BaseVisionModel
from utils.configs import TrainConfig, MetricsConfig
from utils.metrics import Metrics


class VisionTrainer:
    def __init__(self, model: BaseVisionModel, 
                 config: TrainConfig, 
                 train_dataloader: DataLoader, 
                 validation_dataloader: DataLoader,
                 metrics_config: MetricsConfig,
                 resume: Optional[Path] = None) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.metrics = Metrics(metrics_config)

        if resume is not None:
            if not resume.exists():
                raise FileNotFoundError(f"Checkpoint {resume} not found")
            self.load_checkpoint(resume)

        # Initialize wandb for logging
        self.wandb = wandb.init(project="flickr-logo-detection",
                                name=self.model.config.name, 
                                config=self.config,
                                resume="allow",
                                mode="online")

    def get_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer == "SGD":
            return torch.optim.SGD(params, lr=self.config.lr, 
                                   momentum=self.config.momentum, 
                                   weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "Adam":
            return torch.optim.Adam(params, lr=self.config.lr, 
                                    weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config.optimizer} not supported")
    
    def get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.config.scheduler == "StepLR":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, **self.config.scheduler_params)
        elif self.config.scheduler == "ExponentialLR":
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **self.config.scheduler_params)
        else:
            print("No scheduler used")
            return None

    def train(self, output_path: Path) -> None:
        (output_path / "checkpoints").mkdir(exist_ok=True, parents=True)
        self.model.to(self.config.device)
        self.model.train()
        average_losses = {}
        for epoch in range(self.config.epochs):
            pbar = tqdm(self.train_dataloader, total=len(self.train_dataloader))
            for batch_idx, (images, targets) in enumerate(pbar):
                self.optimizer.zero_grad()
                images = list(image.to(self.config.device) for image in images)
                targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                self.optimizer.step()

                if 'train_loss' not in average_losses:
                    average_losses['train_loss'] = losses.item()
                else:
                    average_losses['train_loss'] += losses.item()
                
                for loss_name, loss_value in loss_dict.items():
                    if f"train_{loss_name}" not in average_losses:
                        average_losses[f"train_{loss_name}"] = loss_value.item()
                    else:
                        average_losses[f"train_{loss_name}"] += loss_value.item()

                pbar.set_description(desc=f"Loss: {losses.item()}")
            
            average_losses = {loss_name: loss_value / len(self.train_dataloader) for loss_name, loss_value in average_losses.items()}
            print(f"Epoch: {epoch} - Train Loss: {average_losses['train_loss']}")

            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(epoch, output_path / "checkpoints")

            if self.scheduler is not None:
                self.scheduler.step()


            self.validation(epoch, log_dict={"epoch": epoch, **average_losses, "lr": self.optimizer.param_groups[0]['lr']})

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def save_checkpoint(self, epoch: int, output_dir: Path) -> None:
        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(state_dict, output_dir / f"{self.model.config.name}_{epoch}.pth")

    @torch.no_grad()
    def validation(self, epoch, log_dict: Dict = {}) -> None:
        self.model.train()
        self.metrics.reset()
        average_losses = {}
        pbar = tqdm(self.validation_dataloader, total=len(self.validation_dataloader))
        print(f"Validating at epoch: {epoch}")
        for _, (images, targets) in enumerate(pbar):
            images = list(image.to(self.config.device) for image in images)
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            self.model.eval()
            predictions = self.model(images)
            self.model.train()
            losses = sum(loss for loss in loss_dict.values())

            if 'val_loss' not in average_losses:
                average_losses['val_loss'] = losses.item()
            else:
                average_losses['val_loss'] += losses.item()
            
            for loss_name, loss_value in loss_dict.items():
                if f"val_{loss_name}" not in average_losses:
                    average_losses[f"val_{loss_name}"] = loss_value.item()
                else:
                    average_losses[f"val_{loss_name}"] += loss_value.item()

            self.metrics.update(predictions, targets)

            pbar.set_description(desc=f"Loss: {losses.item()}")
        
        average_losses = {loss_name: loss_value / len(self.validation_dataloader) for loss_name, loss_value in average_losses.items()}
        metrics = self.metrics.compute_and_reset()

        print(f"Epoch: {epoch} - Validation Loss: {average_losses['val_loss']}")
        print(f"Epoch: {epoch} - Validation Metrics: {metrics}")
        self.wandb.log({**average_losses, **metrics, **log_dict})
