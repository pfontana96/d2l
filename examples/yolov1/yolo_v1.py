# Based on Joseph Redmon's YOLOv1 paper
from typing import Union, List

from pydantic import BaseModel, model_validator

from torch import nn
import torchvision as tv


class _ConvLayerConfig(BaseModel):

    model_config = {
        "extra": "forbid"
    }
    
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    maxpool: bool
    padding: Union[int, str]

    def get_torch_layer(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, stride=self.stride),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) if self.maxpool else nn.Identity()
        )


class _LayerConfig(BaseModel):
    model_config = {
        "extra": "forbid"
    }
    
    type: str
    config: _ConvLayerConfig

    @model_validator(mode="after")
    def validate(self) -> "_LayerConfig":

        if self.type != "conv":
            raise ValueError(f"Unsupported layer type: {self.type}")

        return self

    def get_torch_layer(self) -> nn.Sequential:
        return self.config.get_torch_layer()


class YOLOv1(nn.Module):

    _ARCH_CONFIG = [
        {"type": "conv", "config": {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": "same", "maxpool": True}},
        {"type": "conv", "config": {"in_channels": 64, "out_channels": 192, "kernel_size": 3, "stride": 1, "padding": 0, "maxpool": True}},
        {"type": "conv", "config": {"in_channels": 192, "out_channels": 128, "kernel_size": 1, "stride": 1, "padding": 0, "maxpool": False}},
    ]

    def __init__(self):

        self._layers = []
        for layer_config in self._ARCH_CONFIG:
            layer = _LayerConfig(**layer_config).get_torch_layer()
            self._layers.append(layer)

    def prepare_sample(self, x):
        return tv.transforms.Resize((448, 448, 3), interpolation=tv.transforms.InterpolationMode.BILINEAR)(x)

    def forward(self, x):

        x = self.prepare_sample(x)

        for layer in self._layers:
            x = layer(x)

        return x
