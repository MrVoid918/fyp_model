import pytest
import numpy as np
from fyp_model.tracking.association import Association

class TestIOUSep:

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

    @pytest.fixture
    def ext_w_1(self):
        return np.array([[1.2]])

    @pytest.fixture
    def ext_h_1(self):
        return np.array([[0.5]])

    @staticmethod
    def iou_ext_sep(bb_det, bb_trk, ext_w, ext_h):
        """
        Computes extended IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
        with separate extension coefficient
        """
        trk_w = bb_trk[2] - bb_trk[0]
        trk_h = bb_trk[3] - bb_trk[1]
        xx1 = np.maximum(bb_det[0], bb_trk[0] - trk_w*ext_w/2)
        xx2 = np.minimum(bb_det[2], bb_trk[2] + trk_w*ext_w/2)
        w = np.maximum(0., xx2 - xx1)
        if w == 0:
            return 0
        yy1 = np.maximum(bb_det[1], bb_trk[1] - trk_h*ext_h/2)
        yy2 = np.minimum(bb_det[3], bb_trk[3] + trk_h*ext_h/2)
        h = np.maximum(0., yy2 - yy1)
        if h == 0:
            return 0
        wh = w * h
        area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
        area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
        o = wh / (area_det + area_trk - wh)
        return o

    def test_iou_ext_itself(self, box1, ext_w_1, ext_h_1):
        np.testing.assert_allclose(Association.batch_iou_ext_sep(box1, box1, ext_w_1, ext_h_1), 
            TestIOUSep.iou_ext_sep(box1[0], box1[0], ext_w_1, ext_h_1), rtol=1e-5)

    def test_iou_ext_itself_partially_overlapping_boxes(self, box1, box2, ext_w_1, ext_h_1):
        np.testing.assert_allclose(Association.batch_iou_ext_sep(box1, box2, ext_w_1, ext_h_1), 
            TestIOUSep.iou_ext_sep(box1[0], box2[0], ext_w_1, ext_h_1), rtol=1e-5)

    def test_iou_ext_itself_non_overlapping_boxes(self, box1, box3, ext_w_1, ext_h_1):
        np.testing.assert_allclose(Association.batch_iou_ext_sep(box1, box3, ext_w_1, ext_h_1), 
            TestIOUSep.iou_ext_sep(box1[0], box3[0], ext_w_1, ext_h_1), rtol=1e-5)

    def test_compute_iou_fully_contained_boxes(self, box1, box4, ext_w_1, ext_h_1):
        np.testing.assert_allclose(Association.batch_iou_ext_sep(box1, box4, ext_w_1, ext_h_1), 
            TestIOUSep.iou_ext_sep(box1[0], box4[0], ext_w_1, ext_h_1), rtol=1e-3)