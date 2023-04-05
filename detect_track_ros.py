#!/usr/bin/env python3

import argparse
from datetime import datetime
from typing import Union
import time
import os
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from random import randint
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging \
#                 increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# from utils.download_weights import download
from bounding_box_plotter import BoundingBoxPlotter
from mot_converter import MOTConverter
from tracking.sort import Sort
from libs.tracker import Sort_OH

from loguru import logger
from tqdm import tqdm
from typing import Tuple

import cv_bridge
import rospy
from sensor_msgs.msg import Image

def get_ori_imsize(dir: str) -> Tuple[int, int]:
    im_path = list(Path(dir).iterdir())[0]
    img = cv2.imread(str(im_path))
    return img.shape[:2]

class ObjectDetector:

    def __init__(self):
        self.img_size = 640
        self.ori_im = (640, 640)
        self.weights = './weights/yolov7-tiny.pt'
        self.device = select_device('cpu')

        self.model = attempt_load(self.weights, map_location=self.device, )  # load FP32 model
        self.mot_converter = MOTConverter()
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size

        self.bb_plotter = BoundingBoxPlotter()
        self.bridge = cv_bridge.CvBridge()

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if self.half:
            self.model.half()

        # Warm up
        # if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        #     old_img_b = img.shape[0]
        #     old_img_h = img.shape[2]
        #     old_img_w = img.shape[3]
        #     for i in range(3):
        #         model(img, augment=None)[0]

        self.sort = Sort(max_age=5, min_hits=5, iou_threshold=0.25)
        # sort_oh = Sort_OH(max_age=args.max_age, min_hits=args.min_hits, conf_trgt=args.conf_thres, conf_obj=0.75)

    def callback(self, ros_img):
        # Here we call cv_bridge() to convert the ROS image to OpenCV format
        # cv_img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        
        cv_img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        t1 = time_synchronized()
        img = self.preprocess(cv_img)
        with torch.no_grad():
            pred = self.model(img, augment=None)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=None)
        t3 = time_synchronized()

        trackers = []
        # Process detections and track
        for i, det in enumerate(pred, start=1):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_img.shape).round()
                mot_data = self.mot_converter(det, frame = i)
                dets = mot_data[:, 2:7]
                trackers = self.sort.update(dets)

                self.bb_plotter.draw_sort(cv_img, trackers)
                # trackers, unm_tr, unm_gt = sort_oh.update(dets, [])

        cv2.imshow("Image window", cv_img)
        cv2.waitKey(10)

        rospy.loginfo("Done tracking this frame")


    def preprocess(self, cv_img):
        img = letterbox(cv_img, self.img_size, auto= True, stride=self.stride)[0]
        img = img[..., ::-1].transpose(2, 0, 1) # Convert to BGR -> RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

if __name__ == '__main__':
    rospy.init_node('test_node', anonymous=True)

    obj_detector = ObjectDetector()
    rospy.Subscriber("/image_publisher_1680679032077590690/image_raw", Image, obj_detector.callback)

    rospy.spin()
