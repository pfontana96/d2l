import torch
import numpy.testing as npt
import numpy as np

from grund.utils.math import compute_iou, xywh2xyxy


class TestMathUtils:

    def test__given_valid_bboxes__when_compute_iou__then_ok(self):

        # Given
        bbox_a = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
        bbox_b = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)

        # When
        iou = compute_iou(bbox_a, bbox_b)

        # Then
        npt.assert_almost_equal(iou.numpy(force=True), 1/7, decimal=7)

    def test__given_multiple_bboxes_against_one__when_compute_iou__then_ok(self):

        # Given
        bboxes_a = torch.tensor([[1, 1, 3, 3], [1, 2, 3, 4]], dtype=torch.float32)
        bbox_b = bbox_b = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)

        # When
        ious_ab = compute_iou(bboxes_a, bbox_b)
        ious_ba = compute_iou(bbox_b, bboxes_a)

        # Then
        npt.assert_allclose(ious_ab.numpy(force=True), np.array([1/7, 1/3], dtype=np.float32), atol=1e-7)
        npt.assert_allclose(ious_ba.numpy(force=True), np.array([1/7, 1/3], dtype=np.float32), atol=1e-7)

    def test__given_non_intersecting_bboxes__when_compute_iou__then_ok(self):

        # Given
        bbox_a = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
        bbox_b = torch.tensor([[4, 4, 5, 5]], dtype=torch.float32)

        # When
        iou = compute_iou(bbox_a, bbox_b)

        # Then
        npt.assert_almost_equal(iou.flatten().numpy(force=True), 0.0, decimal=7)

    def test__given_valid_bbox__when_xywh2xyxy__then_ok(self):

        # Given
        bbox_xywh = torch.tensor([[2, 2, 2, 2]], dtype=torch.float32)

        # When
        bbox_xyxy = xywh2xyxy(bbox_xywh)

        # Then
        npt.assert_allclose(bbox_xyxy.flatten().numpy(force=True), np.array([1, 1, 3, 3]))