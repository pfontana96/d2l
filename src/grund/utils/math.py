import torch


def xywh2xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x, y, width, height) format to (x1, y1, x2, y2) format.

    Args:
        bbox (torch.Tensor): Bounding box in (x, y, width, height) format.
    Returns:
        torch.Tensor: Bounding box in (x1, y1, x2, y2) format.
    """
    assert bboxes.ndim == 2, f"Expected `bboxes` dimension to be 2, got '{bboxes.ndim}'"
    assert bboxes.shape[1] == 4, f"Expected `bboxes` shape to be (N, 4), got '{bboxes.shape}'"

    new_bbox = torch.empty_like(bboxes, dtype=torch.float32)
    xy = bboxes[:, :2]  # centers
    wh = bboxes[:, 2:] / 2  # half width-height
    new_bbox[:, :2] = xy - wh  # top left xy
    new_bbox[:, 2:] = xy + wh  # bottom right xy

    return new_bbox


def compute_iou(bbox_a_xyxy: torch.Tensor, bbox_b_xyxy: torch.Tensor) -> torch.Tensor:

    assert bbox_a_xyxy.ndim == 2, f"Expected `bbox_a_xyxy` dimension to be 2, got '{bbox_a_xyxy.ndim}'"
    assert bbox_b_xyxy.ndim == 2, f"Expected `bbox_b_xyxy` dimension to be 2, got '{bbox_b_xyxy.ndim}'"

    inter_xy_min = torch.max(bbox_a_xyxy[:, :2], bbox_b_xyxy[:, :2])
    inter_xy_max = torch.min(bbox_a_xyxy[:, 2:], bbox_b_xyxy[:, 2:])
    
    intersections = torch.max(torch.zeros(2), inter_xy_max - inter_xy_min)
    intersections = intersections[:, 0] * intersections[:, 1]

    eps = 1e-6

    a_area = (bbox_a_xyxy[:, 2] - bbox_a_xyxy[:, 0]) * (bbox_a_xyxy[:, 3] - bbox_a_xyxy[:, 1])
    b_area = (bbox_b_xyxy[:, 2] - bbox_b_xyxy[:, 0]) * (bbox_b_xyxy[:, 3] - bbox_b_xyxy[:, 1])

    unions = a_area + b_area - intersections + eps

    iou = intersections / unions

    # Special case with no intersection
    iou[intersections < eps] = 0.0

    return iou
