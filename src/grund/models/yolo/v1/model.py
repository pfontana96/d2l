# Based on Joseph Redmon's YOLOv1 paper
from typing import Union

from pydantic import BaseModel, model_validator

import torch
from torch import nn
import torchvision.transforms.v2 as transforms

from grund.utils.math import xywh2xyxy, compute_iou


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
    def prediction_to_world(tensor: torch.Tensor, width: int, height: int, B: int, C: int) -> torch.Tensor:
        """
        Interprets the network's output for usage

        Args:
            tensor (torch.Tensor): Network output tensor of shape (7, 7, B * 5 + C) where the last dimension stands for
                bounding boxes, defined as: x, y, sqrt(w), sqrt(h), P(obj), P(Ci|obj). Bounding box's center is given
                w.r.t cell's origin while width and height are given w.r.t the input image size. `B` is the number of
                bounding boxes predicted per cell and `C` is the number of classes.

            W (int): Input's image width.
            H (int): Input's image height.
            B (int): Number of bounding boxes per cell.
            C (int): Number of classes

        Returns:
            detections (torch.Tensor): Tensor of shape (49, B, 4 + C) where last dimension stands for the bounding
            boxes coordinates in defined in the input's image grid and the probabilities of each class, i.e.
            (x1, y1, x2, y2, P(C1), ..., P(Cc)).
        """
        assert tensor.shape == (7, 7, B * 5 + C), f"Expected input tensor shape to be (7, 7, B * 5 + C), i.e. {(7, 7, B * 5 + C)}, got {tensor.shape}"  # noqa

        output = torch.empty(49, B, 4 + C, device=tensor.device)

        predicted_bboxes_w_conf = tensor[..., :(B * 5)].reshape(7, 7, B, 5)
        class_conditional_probs = tensor[..., (B * 5):] # shape: (7, 7, C)

        class_probs = predicted_bboxes_w_conf[..., 4].unsqueeze(dim=-1) * class_conditional_probs.unsqueeze(dim=-2)
        class_probs = class_probs.reshape(49, B, C)

        cell_width = width / 7.0
        cell_height = height / 7.0

        xx, yy = torch.meshgrid(
            torch.arange(7, device=tensor.device), torch.arange(7, device=tensor.device), indexing="xy"
        )

        # Extract bbox components
        x =  cell_width * (predicted_bboxes_w_conf[..., 0] + xx.unsqueeze(-1))
        y = cell_height * (predicted_bboxes_w_conf[..., 1] + yy.unsqueeze(-1))
        w = (predicted_bboxes_w_conf[..., 2] ** 2) * width  # network output is square root of width wrt input width
        h = (predicted_bboxes_w_conf[..., 3] ** 2) * height  # network output is square root of height wrt input height

        # Convert to x1, y1, x2, y2
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        output[..., :4] = torch.stack((x1, y1, x2, y2), dim=-1).reshape(49, B, 4)
        output[..., 4:] = class_probs

        return output
    
    @staticmethod
    def loss(
        y_pred: torch.Tensor, y_true: torch.Tensor, lambda_coord: float, lambda_noobj: float,
        number_of_gridcells: int, boxes_per_cell: int, number_of_classes: int
    ) -> torch.Tensor:
        """
        Compute the loss for YOLOv1 predictions.
        
        Args:
        y_pred (torch.Tensor): Tensor of predictions with shape (7, 7, B * 5 + C)
        y_true (torch.Tensor): Tensor of targets with shape (7, 7, B * 5 + C)
        """
        assert y_pred.shape == y_true.shape == (number_of_gridcells, number_of_gridcells, 5*boxes_per_cell + number_of_classes)
        # As stated in the paper:
        # Each gridcell predicts x, y, sqrt(w), sqrt(h), confidence, C1, ..., Cn 
        # where:
        #       confidence = P(obj) * IOU_pred_true
        #       Ci = P(class | obj)
        # At test time, we multiply class conditional probability and the box confidence to get class-based confidence
        # confidence on Ci = P(class | obj) * box's confidence = P(class | obj) * P(obj) * IOU_pred_true = P(class) * IOU_pred_true

        # Get ground truth object mask
        object_presence_mask = y_true[..., 4] > 0
        object_absence_mask = ~object_presence_mask

        # Assume that cell ground truth box is first one
        gt_bboxes_xywh = y_true[object_presence_mask][:4]
        gt_bboxes_xyxy = xywh2xyxy(gt_bboxes_xywh)

        # compute IOU for predictions to find responsible predictor
        # TODO: MOdif iou so that it can broadcast, as we have N,1 gt boxes (one per each image with a detection)
        # but N, B predictor boxes and we need to find the responsable predictor box for each n in N
        
        pred_bboxes_xywhc = y_pred[object_presence_mask].reshape(gt_bboxes_xyxy.shape[0], -1, 5)
        pred_bboxes_xywh = pred_bboxes_xywhc[..., :4]
        pred_bboxes_xyxy = xywh2xyxy(pred_bboxes_xywh)

        ious = torch.empty_like(pred_bboxes_xyxy)
        for i, (gt_bbox, pred_bboxes) in enumerate(zip(gt_bboxes_xyxy, pred_bboxes_xyxy)):
            ious[i, ...] = compute_iou(pred_bboxes, gt_bbox)

        print(f"DEBUG: Computed IOUs:\n{ious.tolist()}")

        # # X and Y loss
        # xy_loss = ((y_true[object_presence_mask][:2] - y_pred[object_presence_mask][:2]) ** 2).sum()

        # # Width and Height loss
        # eps = 1e-6
        # wh_loss = (torch.sqrt(y_true[object_presence_mask][2:] - y_pred[object_presence_mask][2:]) ** 2).sum()

        # # Confidence loss
        # confidence_loss = ((y_true[object_presence_mask][4] - y_pred[object_presence_mask][4]) ** 2 +
        #     lambda_noobj * (y_pred[object_absence_mask][4] ** 2)).sum()
