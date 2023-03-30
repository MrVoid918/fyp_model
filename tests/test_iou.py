import pytest
import numpy as np
from fyp_model.tracking.association import Association

class TestIOU:

    @pytest.fixture
    def box1(self):
        return np.array([[100, 200, 300, 400, -1]])

    @pytest.fixture
    def box2(self):
        return np.array([[200, 300, 400, 500, -1]])

    @pytest.fixture
    def box3(self):
        return np.array([[500, 600, 700, 800, -1]])

    @pytest.fixture
    def box4(self):
        return np.array([[200, 300, 300, 500, -1]])

    def test_iou_itself(self, box1):
        np.testing.assert_allclose(Association.iou_batch(box1, box1), 1.0, rtol=1e-5)

    def test_iou_non_overlapping_boxes(self, box1, box3):
        np.testing.assert_allclose(Association.iou_batch(box1, box3), 0, rtol=1e-5)

    def test_compute_iou_partially_overlapping_boxes(self, box1, box2):
        np.testing.assert_allclose(Association.iou_batch(box1, box2), 0.143, rtol=1e-3)

    def test_compute_iou_fully_contained_boxes(self, box1, box4):
        np.testing.assert_allclose(Association.iou_batch(box1, box4), 0.2, rtol=1e-3)

    def test_iou_two_boxes(self, box1, box2, box3, box4):
        boxes1 = np.vstack((box1, box2))
        boxes2 = np.vstack((box3, box4))

        expected_result = np.array([[0, 0.2], [0, 0.5]])

        np.testing.assert_allclose(Association.iou_batch(boxes1, boxes2), expected_result, rtol=1e-3)