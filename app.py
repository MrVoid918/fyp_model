from PySide2.QtWidgets import (QApplication, QWidget, QLabel, 
    QPushButton, QVBoxLayout)
from PySide2.QtCore import QSize, QTimer, QRect
from PySide2 import QtGui

import qimage2ndarray

import cv2
import sys


class MyApp(QWidget):
    """
    https://gist.github.com/bsdnoobz/8464000?permalink_comment_id=3922087#gistcomment-3922087
    """

    def __init__(self):
        super().__init__()
        self.video_size = QSize(640, 480)
        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        """
        Initialize Widgets
        """
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

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

    def display_video_stream(self):
        """
        Read frame from camera and repaint QLabel widget
        """
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            image = qimage2ndarray.array2qimage(frame)
            pixmap = QtGui.QPixmap.fromImage(image)
            painter = QtGui.QPainter(pixmap)
            pen = QtGui.QPen()
            pen.setColor(QtGui.QColor(255, 0, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(100, 200, 50, 50)
            self.image_label.setPixmap(pixmap)

            painter.end()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())