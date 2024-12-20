import numpy as np
import torch
from numpy.typing import NDArray
from typing import List

from models.osnet import OSNet
from tracking.deepsort.nn_matching import NearestNeighborDistanceMetric
from tracking.deepsort.detection import Detection
from tracking.deepsort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path = 'weights/reid/osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', 
                       model_config=None, 
                       max_dist=0.3, 
                       min_confidence=0.3, 
                       nms_max_overlap=1.0, 
                       max_iou_distance=0.6, 
                       max_age=100, 
                       n_init=3, 
                       nn_budget=100, 
                       use_cuda=True, 
                       use_EMA = False, 
                       use_NSA = False, 
                       use_CMC = False, 
                       cascade_depth=1,):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = OSNet(path=model_path)
        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, 
                               max_iou_distance=max_iou_distance, 
                               max_age=max_age, 
                               n_init=n_init, 
                               use_EMA = use_EMA,
                               use_NSA = use_NSA, 
                               cascade_depth = cascade_depth, 
                               )

    def update(self, 
               det_bbox: NDArray, 
               ori_img: NDArray):
        """
        Perform Update on Data

        Parameters
        ----------
        bbox_xywh : NDArray
            Bounding Boxes in xywh format, 
        ori_img : NDArray
            Original Image
        """
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(det_bbox, ori_img)
        # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        
        detections = [Detection(det_bbox[i, :4], det_bbox[i, 4], features[i, ...]) 
            for i in range(len(det_bbox))]

        # # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_ltrb()
            x1,y1,x2,y2 = box
            # x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, 
                      bbox: NDArray, 
                      ori_img: NDArray) -> NDArray:
        """
        Crop bbox from image and extract features

        Parameters
        ----------
        bbox : NDArray
            Bounding Boxes in xywh format
        ori_img : NDArray
            Original Image

        Returns
        -------
        features : NDArray
            List of Features
        """
        im_crops = []
        H, W, _ = ori_img.shape
        for box in bbox:
            x1 = min(0, int(box[0]))
            y1 = min(0, int(box[1]))
            x2 = max(W, int(x1 + box[2]))
            y2 = max(H, int(y1 + box[3]))
            im = ori_img[y1:y2, x1:x2, :]   # OpenCV: HWC
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops).cpu().numpy()
        else:
            features = np.array([])
        return features