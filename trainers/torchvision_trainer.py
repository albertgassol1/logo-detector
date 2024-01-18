from pathlib import Path

from typing import Optional

import torch
from torch.utils.data import DataLoader

from models import BaseVisionModel
from utils.configs import TrainConfig


class VisionTrainer:
    def __init__(self, model: BaseVisionModel, 
                 config: TrainConfig, 
                 train_dataloader: DataLoader, 
                 validation_dataloader: DataLoader) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.lr, 
                                   momentum=self.config.momentum, 
                                   weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.lr, 
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
        (output_path / "models").mkdir(exist_ok=True, parents=True)
        self.model.to(self.config.device)
        self.model.train()

        for epoch in range(self.config.epochs):
            for batch_idx, (images, targets) in enumerate(self.train_dataloader):
                images = list(image.to(self.config.device) for image in images)
                targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if batch_idx % self.config.log_freq == 0:
                    print(f"Epoch: {epoch} - Batch: {batch_idx} - Loss: {losses}")

            if epoch % self.config.save_freq == 0:
                torch.save(self.model.state_dict(), output_path / "models"/ f"{self.model.config.name}_{epoch}.pth")

            if self.scheduler is not None:
                self.scheduler.step()

            self.validation(epoch)

    @torch.no_grad()
    def validation(self, epoch) -> None:
        self.model.train()
        for batch_idx, (images, targets) in enumerate(self.validation_dataloader):
            images = list(image.to(self.config.device) for image in images)
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            print(f"Epoch: {epoch} - Batch: {batch_idx} - Loss: {losses}")
