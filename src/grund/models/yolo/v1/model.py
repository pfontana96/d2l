# Based on Joseph Redmon's YOLOv1 paper
from typing import Union

from pydantic import BaseModel, model_validator

import torch
from torch import nn
import torchvision.transforms.v2 as transforms


class _ConvLayerConfig(BaseModel):

    model_config = {
        "extra": "forbid"
    }
    
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    maxpool: bool

    def get_torch_layer(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=(self.kernel_size - 1) // 2, stride=self.stride),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) if self.maxpool else nn.Identity()
        )


class _LinearLayerConfig(BaseModel):

    model_config = {
        "extra": "forbid"
    }
    
    in_features: int
    out_features: int
    activate: bool = True

    def get_torch_layer(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.LeakyReLU(0.1, inplace=True) if self.activate else nn.Identity()
        )


class _FlattenLayerConfig(BaseModel):
    def get_torch_layer(self) -> nn.Module:
        return nn.Flatten()


class _LayerConfig(BaseModel):

    model_config = {
        "extra": "forbid"
    }
    
    type: str
    config: Union[_ConvLayerConfig, _LinearLayerConfig, _FlattenLayerConfig]

    @model_validator(mode="after")
    def validate(self) -> "_LayerConfig":

        valid_layer_types = ["conv", "linear", "flatten"]
        if self.type not in valid_layer_types:
            raise ValueError(f"Unsupported layer type: {self.type}, valid types are {valid_layer_types}.")

        return self

    def get_torch_layer(self) -> nn.Module:
        return self.config.get_torch_layer()


class YOLOv1(nn.Module):

    _ARCHITECTURE = [
        # x1
        {"type": "conv", "config": {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "maxpool": True}},
        {"type": "conv", "config": {"in_channels": 64, "out_channels": 192, "kernel_size": 3, "stride": 1, "maxpool": True}},
        {"type": "conv", "config": {"in_channels": 192, "out_channels": 128, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 256, "out_channels": 256, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 1, "maxpool": True}},
        # x 4
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 256, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 256, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 256, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 256, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 1, "maxpool": False}},
        # x1
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 512, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 1024, "kernel_size": 3, "stride": 1, "maxpool": True}},
        # x2
        {"type": "conv", "config": {"in_channels": 1024, "out_channels": 512, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 1024, "kernel_size": 3, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 1024, "out_channels": 512, "kernel_size": 1, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 512, "out_channels": 1024, "kernel_size": 3, "stride": 1, "maxpool": False}},
        # x1
        {"type": "conv", "config": {"in_channels": 1024, "out_channels": 1024, "kernel_size": 3, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 1024, "out_channels": 1024, "kernel_size": 3, "stride": 2, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 1024, "out_channels": 1024, "kernel_size": 3, "stride": 1, "maxpool": False}},
        {"type": "conv", "config": {"in_channels": 1024, "out_channels": 1024, "kernel_size": 3, "stride": 1, "maxpool": False}},
        # Fully connected layer
        {"type": "flatten", "config": {}},
        {"type": "linear", "config": {"in_features": 7 * 7 * 1024, "out_features": 4096}}
    ]

    def __init__(self, boxes_per_cell: int = 2, num_classes: int = 20):
        super().__init__()

        self._boxes_per_cell = boxes_per_cell
        self._num_classes = num_classes

        layers = [_LayerConfig(**layer_config).get_torch_layer() for layer_config in self._ARCHITECTURE]

        # Output layer depends on Model hyper-params
        layers.append(nn.Linear(4096, 7 * 7 * (self._boxes_per_cell * 5 + self._num_classes)))

        self._net = nn.Sequential(*layers)

    def prepare_sample(self, x):
        return transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BILINEAR)(x)

    def forward(self, x):

        x = self.prepare_sample(x)

        return self._net(x).reshape(-1, 7, 7, self._boxes_per_cell * 5 + self._num_classes)
    
    @staticmethod
    def loss(y_pred: torch.Tensor, y_true: torch.Tensor, lambda_coord: float, lambda_noobj: float) -> torch.Tensor:
        """
        Compute the loss for YOLOv1 predictions.
        :param predictions: Model predictions
        :param targets: Ground truth targets
        :return: Computed loss value
        """
        # As stated in the paper:
        # Each gridcell predicts x, y, w, h, confidence, C1, ..., Cn 
        # where:
        #       confidence = P(obj) * IOU_pred_true
        #       Ci = P(class | obj)
        # At test time, we multiply class conditional probability and the box confidence to get class-based confidence
        # confidence on Ci = P(class | obj) * box's confidence = P(class | obj) * P(obj) * IOU_pred_true = P(class) * IOU_pred_true

        # Get ground truth object mask
        object_presence_mask = y_true[..., 4] > 0
        object_absence_mask = ~object_presence_mask

        # X and Y loss
        xy_loss = torch.sum(object_presence_mask * ((y_true[..., 0] - y_pred[..., 0]) ** 2 + (y_true[..., 1] - y_pred[..., 1]) ** 2))

        # Width and Height loss
        wh_loss = torch.sum(
            object_presence_mask * (
                (torch.sqrt(y_true[..., 2]) - torch.sqrt(y_pred[..., 2])) ** 2 + (torch.sqrt(y_true[..., 3]) - torch.sqrt(y_pred[..., 3])) ** 2
            )
        )

        # Confidence loss
        confidence_loss = torch.sum(
            object_presence_mask * ((y_true[..., 4] - y_pred[..., 4]) ** 2) +
            lambda_noobj * object_absence_mask * (y_pred[..., 4] ** 2)
        )
