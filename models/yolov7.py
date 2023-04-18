import argparse
from datetime import datetime
from typing import Union
import time
import os
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from random import randint
from fyp_model.models.experimental import attempt_load
from fyp_model.utils.datasets import LoadStreams, LoadImages
from fyp_model.utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging \
#                 increment_path
# from utils.plots import plot_one_box
from fyp_model.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from fyp_model.utils.datasets import letterbox
# from utils.download_weights import download

class Yolov7Detector:

    def __init__(self, weights: str, img_size: int, device: int):
        self.weights = weights
        self.img_size = img_size
        self.device = select_device(device)

        self.half = self.device.type != 'cpu'
        self.model = attempt_load(weights, map_location=device, )
        self.model.eval()

        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size

        cudnn.benchmark = True  # set True to speed up constant image size inference

        self.model = TracedModel(self.model, self.device, img_size)

        self.model = self.model.half() if self.half else self.model

        if self.device.type != 'cpu':
            # Warmup
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz)
                       .to(self.device)
                       .type_as(next(self.model.parameters())))  # run once
            
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

    def preprocess(self, img0, ) -> torch.Tensor:
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def predict(self, img):
        with torch.no_grad():
            img = self.preprocess(img)
            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=False)
        return img, pred

    def postprocess(self, preds, img, im0s):
        preds = torch.stack(preds, dim=0)[0]
        preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], im0s.shape).round()
        return preds

    def __call__(self, img):
        processed_img, preds = self.predict(img)
        return self.postprocess(preds, processed_img, img)
