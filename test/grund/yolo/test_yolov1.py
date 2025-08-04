import random
import math

import numpy as np
import torch

from grund.models.yolo.v1 import YOLOv1


class TestYOLOv1:

    def test__given_valid_yolo_prediction__when_prediction_to_world__then_ok(self):

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
        interpreted = YOLOv1.prediction_to_world(prediction, image_shape[1], image_shape[0], B, C)

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
