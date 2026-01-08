import random
import math

import numpy as np
import torch

from grund.models.yolo.v1 import YOLOv1


class TestYOLOv1:

    def test__given_valid_yolo_prediction__when_decode__then_ok(self):

        # Given
        B = 2
        C = 5
        image_shape = (480, 640)
        prediction = torch.zeros(7, 7, B * 5 + C)

        cell_width = image_shape[1] / 7
        cell_height = image_shape[0] / 7

        test_cell_x, test_cell_y = [1, 3]  # Cell x=2, Y=4
        row_major_id = test_cell_x + 7 * test_cell_y

        chosen_bbox = random.choice([0, 1])
        x_wrt_cell = 0.5
        y_wrt_cell = 0.3
        w_wrt_image = 1/3
        h_wrt_image = 1/5
        obj_prob = random.choice([0.87, 0.98, 0.67])
        class_conditional_probs = [random.uniform(0.75, 0.95) for _ in range(C)]

        prediction[test_cell_y, test_cell_x][chosen_bbox * 5:chosen_bbox * 5 + 4] = torch.tensor(
            [x_wrt_cell, y_wrt_cell, math.sqrt(w_wrt_image), math.sqrt(h_wrt_image)])
        prediction[test_cell_y, test_cell_x][chosen_bbox * 5 + 4] = obj_prob
        prediction[test_cell_y, test_cell_x][B * 5:] = torch.tensor(class_conditional_probs)

        # When
        interpreted = YOLOv1.decode(prediction, image_shape[1], image_shape[0], B, C)

        # Then
        target_center_x = (test_cell_x + x_wrt_cell) * cell_width
        target_center_y = (test_cell_y + y_wrt_cell) * cell_height
        target_w = w_wrt_image * image_shape[1]
        target_h = h_wrt_image * image_shape[0]

        target_x1 = target_center_x - target_w / 2
        target_y1 = target_center_y - target_h / 2
        target_x2 = target_center_x + target_w / 2
        target_y2 = target_center_y + target_h / 2
        target_class_probs = [p * obj_prob for p in class_conditional_probs]

        np.testing.assert_almost_equal(interpreted[row_major_id, chosen_bbox][:4].numpy(force=True),
                                       np.array([target_x1, target_y1, target_x2, target_y2], dtype=np.float32),
                                       decimal=4)
        np.testing.assert_almost_equal(interpreted[row_major_id, chosen_bbox][4:].numpy(force=True),
                                       np.array(target_class_probs, dtype=np.float32),
                                       decimal=4)
        
    def test__given_valid_bbox__when_encode__then_ok(self):

        # Given
        B = 2
        C = 5
        image_shape = (480, 640)
        x1 = 100
        y1 = 150
        x2 = 200
        y2 = 250
        class_probs = [random.uniform(0.75, 0.95) for _ in range(C)]
        bbox = torch.tensor([[x1, y1, x2, y2, *class_probs]], dtype=torch.float32)

        cell_width = image_shape[1] / 7
        cell_height = image_shape[0] / 7

        # When
        encoded = YOLOv1.encode(bbox, image_shape[1], image_shape[0], B, C)

        # Then
        expected_shape = (7, 7, B * 5 + C)
        assert encoded.shape == expected_shape, f"Expected shape {expected_shape}, got {encoded.shape}"

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        x_wrt_cell = (x_center / cell_width) % 1
        y_wrt_cell = (y_center / cell_height) % 1

        x_cell_id = int(x_center // cell_width)
        y_cell_id = int(y_center // cell_height)

        w = (x2 - x1) / image_shape[1]
        h = (y2 - y1) / image_shape[0]

        expected_encoded = np.zeros((7, 7, B * 5 + C), dtype=np.float32)
        expected_encoded[y_cell_id, x_cell_id, :5] = np.array(
            [x_wrt_cell, y_wrt_cell, math.sqrt(w), math.sqrt(h), 1.0], dtype=np.float32)
        expected_encoded[y_cell_id, x_cell_id, B * 5:] = np.array(
            class_probs, dtype=np.float32
        )

        np.testing.assert_almost_equal(
            encoded.numpy(force=True), expected_encoded, decimal=4,
            err_msg=f"Encoded tensor does not match expected values at cell ({x_cell_id}, {y_cell_id})"
        )

    def test__given_multiple_valid_bboxes_for_same_cell__when_encode__then_keep_1(self):
        
        # Given
        B = 2
        C = 5
        image_shape = (480, 640)

        cell_width = image_shape[1] / 7
        cell_height = image_shape[0] / 7

        x_cell_id = random.randint(0, 6)
        y_cell_id = random.randint(0, 6)

        x11_wrt_cell = random.uniform(0, 0.45)
        y11_wrt_cell = random.uniform(0, 0.45)
        x12_wrt_cell = random.uniform(0.55, 1.0)
        y12_wrt_cell = random.uniform(0.55, 1.0)

        x11 = cell_width * (x_cell_id + x11_wrt_cell)
        y11 = cell_height * (y_cell_id + y11_wrt_cell)
        x12 = cell_width * (x_cell_id + x12_wrt_cell)
        y12 = cell_height * (y_cell_id + y12_wrt_cell)

        x21_wrt_cell = random.uniform(0, 0.45)
        y21_wrt_cell = random.uniform(0, 0.45)
        x22_wrt_cell = random.uniform(0.55, 1.0)
        y22_wrt_cell = random.uniform(0.55, 1.0)

        x21 = cell_width * (x_cell_id + x21_wrt_cell)
        y21 = cell_height * (y_cell_id + y21_wrt_cell)
        x22 = cell_width * (x_cell_id + x22_wrt_cell)
        y22 = cell_height * (y_cell_id + y22_wrt_cell)

        bboxes = torch.tensor([
            [x11, y11, x12, y12, *([0.8] * C)],
            [x21, y21, x22, y22, *([0.9] * C)]
        ], dtype=torch.float32)

        # When
        encoded = YOLOv1.encode(bboxes, image_shape[1], image_shape[0], B, C)

        # Then
        expected_shape = (7, 7, B * 5 + C)
        assert encoded.shape == expected_shape, f"Expected shape {expected_shape}, got {encoded.shape}"

        x_center = (x11 + x12) / 2
        y_center = (y11 + y12) / 2

        x_wrt_cell = (x_center / cell_width) % 1
        y_wrt_cell = (y_center / cell_height) % 1

        w = (x12 - x11) / image_shape[1]
        h = (y12 - y11) / image_shape[0]

        expected_encoded = np.zeros((7, 7, B * 5 + C), dtype=np.float32)
        expected_encoded[y_cell_id, x_cell_id, :5] = np.array(
            [x_wrt_cell, y_wrt_cell, math.sqrt(w), math.sqrt(h), 1.0], dtype=np.float32)
        expected_encoded[y_cell_id, x_cell_id, B * 5:] = bboxes[0, 4:].numpy(force=True)

        np.testing.assert_allclose(
            encoded.float().numpy(force=True), expected_encoded, rtol=1e-4, atol=1e-4,
            err_msg=f"Encoded tensor does not match expected values at cell ({x_cell_id}, {y_cell_id})"
        )