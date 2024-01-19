from abc import abstractmethod
import torch.nn as nn
from torchvision.models import detection

from typing import Dict

from utils.configs import ModelConfig

class BaseVisionModel(nn.Module):
    methods: Dict[str, Dict[str, str]] = dict()
    method2class: Dict[str, nn.Module] = dict()
    method : Dict[str, str] = dict()

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        assert self.config.name in self.methods.keys(), f"Method {self.config.name} not supported"
        assert self.config.num_classes > 0, "Number of classes must be greater than 0"

    def __init_subclass__(cls):
        '''Register the child classes into the parent'''
        name = cls.method['name']
        cls.methods[name] = cls.method
        cls.method2class[name] = cls

    def __new__(cls, config: ModelConfig, *_, **__):
        '''Instanciate the object from the child class'''
        return super().__new__(cls.method2class[config.name])

    @abstractmethod
    def forward(self, image, target = None):
        pass


class FRCNN(BaseVisionModel):
    method = {'name': 'FRCNN'}

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
        # self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Freeze backbone
        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, 
                                                                                     config.num_classes)
        
    def forward(self, image, target = None):
        # Returns losses if training and prediction if inferencing
        return self.model(image, target)


class SSD(BaseVisionModel):
    method = {'name': 'SSD'}

    def __init__(self, config: ModelConfig, size: int = 300):
        super().__init__(config)
        self.model = detection.ssd300_vgg16(weights=detection.SSD300_VGG16_Weights.COCO_V1)
        in_channels = detection._utils.retrieve_out_channels(self.model.backbone, (size, size))
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        # The classification head.
        self.model.head.classification_head = detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=self.config.num_classes,
        )
        # Image size for transforms.
        self.model.transform.min_size = (size,)
        self.model.transform.max_size = size
        
        # Freeze backbone
        if config.freeze:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, image, target = None):
        return self.model(image, target)
