# Based on Joseph Redmon's YOLOv1 paper
from pathlib import Path
from typing import Union, Self, Tuple, List

from pydantic import BaseModel, model_validator, ValidationError

import torch
from torch import nn
import torchvision.transforms.v2 as transforms

from grund.utils.math import xywh2xyxy, compute_iou, limit_row_repeats_torch
from grund.models.yolo.v1.layers_config import ArchitectureConfig


class YOLOv1Params(BaseModel):

    model_config = {
        "extra": "forbid"
    }

    B: int
    S: int
    C: int

    device: str = "cpu"

    @model_validator(mode="after")
    def validate(self) -> Self:
        available_devices = ["cuda", "cpu"]
        if not self.device in available_devices:
            raise ValidationError(f"Expected device to be one of {available_devices}, got '{self.device}'")
        
        return self

    def model_post_init(self, context):

        self._torch_device = torch.device(self.device)

        self._xx, self._yy = torch.meshgrid(
            torch.arange(self.S, device=self.device), torch.arange(self.S, device=self.device), indexing="xy"
        )

        self._image_shape = (int(self.S * 64), int(self.S * 64))

        self._cell_width = self._image_shape[1] / self.S
        self._cell_height = self._image_shape[0] / self.S

    @property
    def torch_device(self) -> torch.device:
        return self._torch_device
    
    @property
    def grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self._xx, self._yy)
    
    @property
    def img_shape(self) -> Tuple[int, int]:
        return self._image_shape
    
    @property
    def cell_size(self) -> Tuple[float, float]:
        """
            Returns cell's size

        Returns:
            Tuple[float, float]: (cell_height, cell_width).
        """
        return (self._cell_height, self._cell_width)
  

class YOLOv1(nn.Module):

    _ARCHITECTURE_CONFIG = Path(__file__).parent / "arch.yaml"

    def __init__(self, yolo_params: Union[dict, YOLOv1Params] = YOLOv1Params(B=2, S=7, C=20)):

        super().__init__()

        if isinstance(yolo_params, dict):
            yolo_params = YOLOv1Params(**yolo_params)

        self._params = yolo_params

        arch = ArchitectureConfig.from_yaml(self._ARCHITECTURE_CONFIG)

        layers = [layer_config.get_torch_layer() for layer_config in arch.layers]

        # Output layers depend on Model hyper-params
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=(self._params.S * self._params.S * 1024), out_features=4096))
        layers.append(nn.Linear(4096, self._params.S * self._params.S * (self._params.B * 5 + self._params.C)))

        self._net = nn.Sequential(*layers)

    @property
    def params(self) -> YOLOv1Params:
        return self._params

    def prepare_sample(self, x):
        return transforms.Resize(self._params.img_shape, interpolation=transforms.InterpolationMode.BILINEAR)(x)

    def forward(self, x):

        x = self.prepare_sample(x)
        return self._net(x).reshape(-1, self._params.S, self._params.S, self._params.B * 5 + self._params.C)
    
    @staticmethod
    def decode(tensor: torch.Tensor, params: YOLOv1Params, keep_dim: bool = False) -> torch.Tensor:
        """
        Interprets the network's output for usage

        Args:
            tensor (torch.Tensor): Network output tensor of shape (S, S, B * 5 + C) where the last dimension stands for
                bounding boxes, defined as: x, y, sqrt(w), sqrt(h), P(obj), P(Ci|obj). Bounding box's center is given
                w.r.t cell's origin while width and height are given w.r.t the input image size. `B` is the number of
                bounding boxes predicted per cell and `C` is the number of classes.

            params (YOLOv1Params): Class container YOLO params (S, B, C)
            keep_dim (bool): If True, the output will have shape (S, S, B, 4 + C). If False, the output will have shape (S * S, B, 4 + C) where the last dimension stands for the bounding boxes coordinates

        Returns:
            detections (torch.Tensor): Tensor of shape (S * S, B, 4 + C) where last dimension stands for the bounding
            boxes coordinates in defined in the input's image grid and the probabilities of each class, i.e.
            (x1, y1, x2, y2, P(C1), ..., P(Cc)).
        """
        assert tensor.shape == (params.S, params.S, params.B * 5 + params.C), f"Expected input tensor shape to be (S, S, B * 5 + C), i.e. {(params.S, params.S, params.B * 5 + params.C)}, got {tensor.shape}"  # noqa

        output = torch.empty(params.S * params.S, params.B, 4 + params.C, device=params.torch_device)

        predicted_bboxes_w_conf = tensor[..., :(params.B * 5)].reshape(params.S, params.S, params.B, 5)
        class_conditional_probs = tensor[..., (params.B * 5):] # shape: (S, S, C)

        class_probs = predicted_bboxes_w_conf[..., 4].unsqueeze(dim=-1) * class_conditional_probs.unsqueeze(dim=-2)
        class_probs = class_probs.reshape(params.S * params.S, params.B, params.C)

        height, width = params.img_shape
        cell_height, cell_width = params.cell_size

        xx, yy = params.grid

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

        output[..., :4] = torch.stack((x1, y1, x2, y2), dim=-1).reshape(49, params.B, 4)
        output[..., 4:] = class_probs

        return output if not keep_dim else output.reshape(params.S, params.S, params.B, 4 + params.C)

    @staticmethod
    def encode(tensor: torch.Tensor, params: YOLOv1Params, as_yolo: bool = True) -> torch.Tensor:
        """
        Encodes the bounding boxes and class probabilities into the network's output format.

        Args:
            tensor (torch.Tensor): Tensor of shape (N, 4 + C) where last dimension stands for the bounding
                boxes coordinates in defined in the input's image grid and the probabilities of each class, i.e.
                (x1, y1, x2, y2, P(C1), ..., P(Cc)). and N is the number of detection to encode (at most (S * S) * B).
            params (YOLOv1Params): Class container YOLO params (S, B, C).
            as_yolo (bool): If True, the output will be in YOLO format. If False, the output bounding boxes will be
                in (x1, y1, x2, y2) format and class probabilities will not be conditionals.

        Returns:
            encoded_tensor (torch.Tensor): Tensor of shape (S, S, B * 5 + C) where the last dimension stands for
                bounding boxes, defined as: x, y, sqrt(w), sqrt(h), P(obj), P(Ci|obj) if as_yolo is True, otherwise
        """
        assert tensor.ndim == 2, f"Expected input tensor shape to be 2, got '{tensor.ndim}'"
        assert tensor.shape[0] <= (params.S * params.S) * params.B, f"Expected input tensor to have at most {params.S * params.S * params.B} detections, got {tensor.shape[0]}"
        assert tensor.shape[1] == 4 + params.C, f"Expected input tensor to have {4 + params.C} channels, got {tensor.shape[1]}"

        encoded_tensor = torch.zeros((params.S, params.S, params.B * 5 + params.C), device=tensor.device) if as_yolo else torch.zeros((params.S, params.S, 4 + params.C), device=tensor.device)

        # Compute bounding boxes in YOLOv1 format
        x1, y1, x2, y2, class_probs = tensor[..., 0], tensor[..., 1], tensor[..., 2], tensor[..., 3], tensor[..., 4:]
        w = x2 - x1
        h = y2 - y1

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        height, width = params.img_shape
        cell_height, cell_width = params.cell_size

        x = x_center / cell_width
        y = y_center / cell_height

        x_cell_id = torch.floor(x).long()
        y_cell_id = torch.floor(y).long()

        x_wrt_cell = x - x_cell_id.float()
        y_wrt_cell = y - y_cell_id.float()

        # Ensure at most 1 boxes per cell
        mask = limit_row_repeats_torch(torch.stack((x_cell_id, y_cell_id), dim=-1), max_repeats=1)

        x_cell_id = x_cell_id[mask]
        y_cell_id = y_cell_id[mask]

        if as_yolo:
            x_wrt_cell = x_wrt_cell[mask]
            y_wrt_cell = y_wrt_cell[mask]
            w = torch.sqrt(w[mask] / width)  # network output is square root
            h = torch.sqrt(h[mask] / height)  # network output is square root

            # Fill the encoded tensor
            encoded_tensor[y_cell_id, x_cell_id, :5] = torch.stack(
                (x_wrt_cell, y_wrt_cell, w, h, torch.ones_like(x_wrt_cell)), dim=-1
            )

            encoded_tensor[y_cell_id, x_cell_id, -params.C:] = class_probs[mask]

        else:
            x1 = x1[mask]
            y1 = y1[mask]
            x2 = x2[mask]
            y2 = y2[mask]

            encoded_tensor[y_cell_id, x_cell_id, :4] = torch.stack((x1, y1, x2, y2), dim=-1)
            encoded_tensor[y_cell_id, x_cell_id, 4:] = class_probs[mask]

        return encoded_tensor

    @staticmethod
    def loss(
        y_pred: torch.Tensor, y_true: torch.Tensor, lambda_coord: float, lambda_noobj: float, params: YOLOv1Params
    ) -> torch.Tensor:
        """
        Compute the loss for YOLOv1 predictions.
        
        Args:
            y_pred (torch.Tensor): Tensor of predictions with shape (7, 7, B * 5 + C)
            y_true (torch.Tensor): Tensor of targets with shape (N, 4 + C) where N is the number of detections.
            lambda_coord (float): Weight for the coordinate loss.
            lambda_noobj (float): Weight for the no-object confidence loss.
            params (YOLOv1Params): Class container YOLO params (S, B, C).
        """
        # As stated in the paper:
        # Each gridcell predicts x, y, sqrt(w), sqrt(h), confidence, C1, ..., Cn 
        # where:
        #       confidence = P(obj) * IOU_pred_true
        #       Ci = P(class | obj)
        # At test time, we multiply class conditional probability and the box confidence to get class-based confidence
        # confidence on Ci = P(class | obj) * box's confidence = P(class | obj) * P(obj) * IOU_pred_true = P(class) * IOU_pred_true

        y_true_ = YOLOv1.encode(y_true, params, as_yolo=False)
        y_true_as_yolo = YOLOv1.encode(y_true, params, as_yolo=True)
        y_pred_decoded = YOLOv1.decode(y_pred, params, keep_dim=True)

        # Get ground truth object mask
        object_ids_y, object_ids_x = torch.where(y_true_[..., 4] > 0)
        bbox_responsible_ids = torch.empty_like(object_ids_y)
        # object_absence_mask = ~object_presence_mask

        # Compute IOU for predictions to find responsible predictor
        for i, (pred_bboxes, gt_bboxes) in enumerate(zip(y_pred_decoded[(object_ids_y, object_ids_x)], y_true_[(object_ids_y, object_ids_x)])):
            bbox_responsible_ids[i] = compute_iou(pred_bboxes[..., :4], gt_bboxes[:4].unsqueeze(dim=0)).argmax(dim=-1)

        squared_error = (y_true_as_yolo[object_ids_y, object_ids_x, :5] - y_pred[object_ids_y, object_ids_x, (bbox_responsible_ids * 5):(bbox_responsible_ids * 5 + 5)]) ** 2

        # XYsqrt(w)sqrt(h) loss
        xywh_loss = squared_error[..., :4].sum()

        # Confidence loss
        conf_loss = squared_error[..., 4].sum()

        print(f"DEBUG: xywh_loss: {xywh_loss} and conf loss: {conf_loss}")
        