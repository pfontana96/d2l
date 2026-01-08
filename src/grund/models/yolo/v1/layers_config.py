from pathlib import Path
from typing import Union, List

import yaml
from pydantic import BaseModel, model_validator

from torch import nn


class ArchitectureConfig(BaseModel):
    
    model_config = {
        "extra": "forbid"
    }

    B: int
    S: int
    C: int

    layers: List["LayerConfig"]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ArchitectureConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if not path.suffix == ".yaml":
            raise ValueError(f"Configuration file must be a YAML file: {path}")
        
        with path.open("r") as file:
            config_data = yaml.safe_load(file)

        return cls(**config_data)


class LayerConfig(BaseModel):

    model_config = {
        "extra": "forbid"
    }
    
    type: str
    config: Union["_ConvLayerConfig", "_LinearLayerConfig", "_FlattenLayerConfig"]

    @model_validator(mode="after")
    def validate(self) -> "LayerConfig":

        valid_layer_types = ["conv", "linear", "flatten"]
        if self.type not in valid_layer_types:
            raise ValueError(f"Unsupported layer type: {self.type}, valid types are {valid_layer_types}.")

        return self

    def get_torch_layer(self) -> nn.Module:
        return self.config.get_torch_layer()


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
