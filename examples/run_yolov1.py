import argparse
from pathlib import Path
import logging
from typing import Union

from PIL import Image
from pydantic import BaseModel, model_validator

import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageNet

from d2l_pep.log import set_root_logger

from yolo_v1.model import YOLOv1


logger = logging.getLogger(__name__)


class _TrainConfig(BaseModel):

    model_config = {
        "extra": "forbid"
    }

    imagenet_path: Union[str, Path]

    @model_validator(mode='after')
    def validate(self):
        if isinstance(self.imagenet_path, str):
            self.imagenet_path = Path(self.imagenet_path)

        if not self.imagenet_path.is_dir():
            raise ValueError(f"Imagenet path {self.imagenet_path} is not a valid directory.")

        return self


def parse_args():

    parser = argparse.ArgumentParser(description="YOLOv1 CLI")

    subparsers = parser.add_subparsers(dest='action', help='Available actions')

    # Train parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("config_file", type=str, help="Path to the training configuration file")

    # Predict parser
    predict_parser = subparsers.add_parser("pred")
    predict_parser.add_argument("image_path", type=str, help="Path to the image to predict")
    predict_parser.add_argument("--weights", type=str, default=None, help="Path to the model weights file")

    args = parser.parse_args()

    if args.action not in ["train", "pred"]:
        parser.error("Invalid action. Use 'train' or 'pred'.")

    return args


def main():

    set_root_logger(True)

    args = parse_args()

    model = YOLOv1()

    if args.action == "train":
        pass

    elif args.action == "pred":
        image_path = Path(args.image_path)
        if not image_path.is_file():
            raise ValueError(f"Image path {image_path} does not point to a valid file.")
        
        image = Image.open(image_path).convert("RGB")
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)  # Add batch dimension

        logger.info(f"Loaded image: {image_path} (shape: {image.shape})")

        output = model.forward(image)

        logger.info(f"Prediction output: {output.shape}")


if __name__ == "__main__":
    main()