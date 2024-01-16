from abc import abstractmethod
import torch.nn as nn
import torch
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
    def forward(self, x):
        pass


class FRCNN(BaseVisionModel):
    def __init__(self, config):
        super().__init__(config)
        weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


class SSD(BaseVisionModel):
    def __init__(self, config):
        super().__init__(config)
        weights = detection.SSD300_VGG16_Weights.DEFAULT
        self.model = detection.ssd300_vgg16(weights=weights)


        
        
        


    def forward(self, x):
        return self.model(x)
