import torch


def xywh2xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box from (x, y, width, height) format to (x1, y1, x2, y2) format.

    Args:
        bbox (torch.Tensor): Bounding box in (x, y, width, height) format.
    Returns:
        torch.Tensor: Bounding box in (x1, y1, x2, y2) format.
    """

    new_bbox = torch.empty_like(bbox, dtype=torch.float32)
    xy = new_bbox[..., :2]  # centers
    wh = new_bbox[..., 2:] / 2  # half width-height
    new_bbox[..., :2] = xy - wh  # top left xy
    new_bbox[..., 2:] = xy + wh  # bottom right xy

    return new_bbox


def iou(bbox_a_xyxy: torch.Tensor, bbox_b_xyxy: torch.Tensor):

    xyA = torch.max(bbox_a_xyxy[:2], bbox_b_xyxy[:2])
    xyB = torch.min(bbox_a_xyxy[3:], bbox_b_xyxy[3:])
