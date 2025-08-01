import argparse
from pathlib import Path
import logging

from PIL import Image

from d2l_pep.log import set_root_logger

from .yolo_v1 import YOLOv1


logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="YOLOv1 CLI")
    parser.add_argument("action", choices=["train", "predict"], help="Action to perform: train or predict")

    # Train parser
    train_parser = parser.add_argument_group("Train options")
    train_parser.add_argument("config_file", type=str, help="Path to the training configuration file")

    # Predict parser
    predict_parser = parser.add_argument_group("Predict options")
    predict_parser.add_argument("image_path", type=str, help="Path to the image to predict")
    predict_parser.add_argument("--weights", type=str, default=None, help="Path to the model weights file")

    return parser.parse_args()


def main():

    set_root_logger(True)

    args = parse_args()

    model = YOLOv1()

    if args.action == "train":
        pass

    elif args.action == "predict":
        image_path = Path(args.image_path)
        if not image_path.is_file():
            raise ValueError(f"Image path {image_path} does not point to a valid file.")
        
        image = Image.open(image_path).convert("RGB")
        output = model.forward(image)

        logger.info(f"Prediction output: {output}")


if __name__ == "__main__":
    main()