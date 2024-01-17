from abc import abstractmethod
import torch.nn as nn
from torchvision.models import detection

from typing import Dict

class BaseVisionModel(nn.Module):
    methods: Dict[str, Dict[str, str]] = dict()
    method2class: Dict[str, nn.Module] = dict()
    method : Dict[str, str] = dict()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __init_subclass__(cls):
        '''Register the child classes into the parent'''
        name = cls.method['name']
        cls.methods[name] = cls.method
        cls.method2class[name] = cls

    def __new__(cls, config, *_, **__):
        '''Instanciate the object from the child class'''
        return super().__new__(cls.method2class[config['name']])

    @abstractmethod
    def forward(self, image, target = None):
        pass


class FRCNN(BaseVisionModel):
    def __init__(self, config):
        super().__init__(config)
        weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

        # Freeze backbone
        if config['freeze']:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, 
                                                                                     config['num_classes'])
        
    def forward(self, image, target = None):
        # Returns losses if training and prediction if inferencing
        return self.model(image, target)


class SSD(BaseVisionModel):
    def __init__(self, config):
        super().__init__(config)
        weights = detection.SSD300_VGG16_Weights.DEFAULT
        self.model = detection.ssd300_vgg16(weights=weights)
        
        # Freeze backbone
        if config['freeze']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, image, target = None):
        return self.model(image, target)
