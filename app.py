from PySide2.QtWidgets import (QApplication, QWidget, QLabel, 
    QPushButton, QVBoxLayout)
from PySide2.QtCore import QSize, QTimer, QRect, Slot
from PySide2 import QtGui

import qimage2ndarray

import numpy as np
import cv2
import torch
import sys
from loguru import logger

from models.yolov7 import Yolov7Detector
from models.traj_predictor import PredictTrajectory
from models.cmc import CMC
from tracking.sort import Sort
from bounding_box_plotter import BoundingBoxPlotter
from mot_converter import MOTConverter

class MyApp(QWidget):
    """
    https://gist.github.com/bsdnoobz/8464000?permalink_comment_id=3922087#gistcomment-3922087
    """

    def __init__(self):
        super().__init__()
        self.video_size = QSize(640, 480)
        self.frame = 0
        self.tracking_data = None
        self.prev_frame = None
        self.setup_ui()
        self.setup_model()
        self.setup_camera()

    def setup_ui(self):
        """
        Initialize Widgets
        """
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.quit_button)
        self.main_layout.addWidget(self.predict_button)

        self.setLayout(self.main_layout)

    @Slot()
    def predict(self):
        # TODO: Implement prediction of trajectory
        pass

    def setup_model(self):
        assert torch.cuda.is_available(), "CUDA unavailable, invalid device 0 requested"
        self.model = Yolov7Detector(weights="weights/yolov7-tiny.pt", img_size=320, device="cuda:0")
        self.tracker = Sort(max_age=10, min_hits=10)
        self.mot_converter = MOTConverter()
        self.bb_plotter = BoundingBoxPlotter()
        self.history = 60
        self.traj_predictor = PredictTrajectory(
            model_path="weights/pred_22_03_23_1632.tar",
            img_height=1080,
            img_width=1920,
            past_frames=60, 
            future_frames=30, 
            input_size = 8, 
            enc_hidden_size = 512, 
            enc_output_size = 256, 
            velocity=True, 
        ).cuda()
        self.cmc = CMC(strategy="ECC")

    def setup_camera(self):
        """
        Initialize Camera
        """
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    @Slot()
    def predict(self):
        self.predicted_data = self.traj_predictor.predict(
            data = self.tracking_data, 
            person_id = 1, 
            frame_id = 0, )

    def update_tracking_data(self, tracking_data):
        if self.tracking_data is None:
            self.tracking_data = tracking_data
        else:
            self.tracking_data = np.vstack((self.tracking_data, tracking_data))
        self.tracking_data = self.tracking_data[np.where(self.tracking_data[:, 0] >= self.frame - self.history)]

    def display_video_stream(self):
        """
        Read frame from camera and repaint QLabel widget
        """
        ret, frame = self.capture.read()
        if ret:
            if self.prev_frame is not None:
                pass
                # ecc = self.cmc(frame, self.prev_frame)
            self.prev_frame = frame
            # --------- Detection --------- #
            preds = self.model(frame)
            mot_data = self.mot_converter(preds, frame = self.frame)
            dets = mot_data[:, 2:7]
            # --------- Tracking --------- #
            trackers = self.tracker.update(dets)
            self.update_tracking_data(self.mot_converter.sort_to_mot(trackers, frame = self.frame))
            # --------- Plotting --------- #
            logger.info(f"Tracking_Data: {self.tracking_data.shape}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.bb_plotter.draw_sort(frame, trackers, id = True)
            # frame = cv2.flip(frame, 1)
            image = qimage2ndarray.array2qimage(frame)
            pixmap = QtGui.QPixmap.fromImage(image)
            painter = QtGui.QPainter(pixmap)
            pen = QtGui.QPen()
            pen.setColor(QtGui.QColor(255, 0, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            # drawRect(x, y, width, height)
            # painter.drawRect(100, 200, 50, 50)
            self.image_label.setPixmap(pixmap)

            painter.end()
            self.frame += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())