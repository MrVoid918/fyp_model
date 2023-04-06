import numpy as np
from pathlib import Path

import cv2 as cv
import random

class BoundingBoxPlotter():

    def __init__(self) -> None:
        self.classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traff`ic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
        ]
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(300)]

    def _plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        # bbox is xyxy format
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled
            cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)

    def draw(self, img, boxinfo):
        for xyxy, conf, cls in boxinfo:
            self._plot_one_box(xyxy, img, label=self.classes[int(cls)], color=self.colors[int(cls)], line_thickness=2)
        cv.imshow('Press ESC to Exit', img) 
        cv.waitKey(5000)

    def draw_yolov7(self, img, boxinfo):
        for box in boxinfo:
            self._plot_one_box(box[:4], img, label=self.classes[int(box[-1])], color=self.colors[int(box[-1])], line_thickness=2)
        cv.imshow('Press ESC to Exit', img) 
        cv.waitKey(5000)

    def draw_sort(self, img, boxinfo, id = True):
        """
        Draw bounding boxes on image based on SORT output
        :param img: image to draw on
        :param boxinfo: list of bounding boxes in SORT format [[frame, x, y, x, y, 1, -1, -1, -1]]
        """
        for box in boxinfo:
            # don't plot if there is no bounding box, just return the image
            if np.any(box):
                if id:
                    self._plot_one_box(box[:4], img, label=str(int(box[4])), color=self.colors[int(box[4]) % 300], line_thickness=2)
                else:
                    self._plot_one_box(box[:4], img, label=self.classes[0], color=self.colors[int(box[4]) % 300], line_thickness=2)

    def draw_mot(self, img, boxinfo, label = None, color = None) -> None:
        """
        Draw bounding boxes on image based on MOT challenge format
        Does not return images, rather it overwrites the images with the bounding boxes drawn on them
        :param img: image to draw on
        :param boxinfo: list of bounding boxes in MOT format [[frame, id, x, y, w, h, 1, -1, -1, -1]]
        """
        for box in boxinfo:
            # don't plot if there is no bounding box, just return the image
            if np.any(box):
                box[4:6] = box[4:6] + box[2:4]
                if label is not None:
                    box_label = label
                else:
                    box_label = self.classes[0]

                if color is not None:
                    box_color = color
                else:
                    box_color = self.colors[int(box[1]) % 300]

                self._plot_one_box(box[2:6], img, label=box_label, color=box_color, line_thickness=2)