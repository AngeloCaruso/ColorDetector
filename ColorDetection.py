import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QColorDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np


class ColorDetector(QDialog):
    def __init__(self):
        super(ColorDetector, self).__init__()
        loadUi('resources/OpenCV.ui', self)
        self.image = None
        self.timerImg = None
        self.timerVideo = None
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_webcam)
        self.load_button.clicked.connect(self.load_image)
        self.color1_button.clicked.connect(self.set_color1)
        self.color_picker.clicked.connect(self.pick_color)
        # self.color2_button.clicked.connect(self.set_color2)

    def start_webcam(self):
        if self.timerImg is not None:
            self.timerImg.stop()

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

        self.timerVideo = QTimer(self)
        self.timerVideo.timeout.connect(self.update_frame_video)
        self.timerVideo.start(5)

    def stop_webcam(self):
        self.capture.release()
        self.timerVideo.stop()

    def load_image(self):
        if self.timerVideo is not None:
            self.timerVideo.stop()

        imgRoute = QFileDialog.getOpenFileName(self, 'Open file')
        self.image = cv2.imread(imgRoute[0])
        self.timerImg = QTimer(self)
        self.timerImg.timeout.connect(self.update_frame_img)
        self.timerImg.start(5)

    def set_color1(self):
        self.color1_lower = np.array([self.h_min.value(), self.s_min.value(), self.v_min.value()], np.uint8)
        self.color1_upper = np.array([self.h_max.value(), self.s_max.value(), self.v_max.value()], np.uint8)

        self.color1_value.setText(' Min : ' + str(self.color1_lower) + ' Max: ' + str(self.color1_upper))

    def set_color2(self):
        self.color2_lower = np.array([self.h_min.value(), self.s_min.value(), self.v_min.value()], np.uint8)
        self.color2_upper = np.array([self.h_max.value(), self.s_max.value(), self.v_max.value()], np.uint8)

        self.color2_value.setText('C2 -> Min :' + str(self.color2_lower) + ' Max: ' + str(self.color2_upper))

    def pick_color(self):
        color = QColorDialog.getColor()
        h = color.getHsv()[0] / 2
        s = color.getHsv()[1]
        v = color.getHsv()[2]

        self.h_min.setValue(h - 5)
        self.s_min.setValue(s)
        self.v_min.setValue(255 - v)

        self.h_max.setValue(h + 5)
        self.s_max.setValue(255)
        self.v_max.setValue(255)

    def update_frame_video(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.displayImage(self.image, 1)
        # Reference
        # lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} #assign new item lower['blue'] = (93, 10, 0)
        # upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}

        blurred = cv2.blur(self.image, (7, 7))
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        color_lower = np.array([self.h_min.value(), self.s_min.value(), self.v_min.value()], np.uint8)
        color_upper = np.array([self.h_max.value(), self.s_max.value(), self.v_max.value()], np.uint8)

        self.current_value.setText('Current Value -> Min :' + str(color_lower) + ' Max: ' + str(color_upper))

        mask = cv2.inRange(hsv, color_lower, color_upper)
        dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))
        erode = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

        color_mask = cv2.erode(mask, erode)
        color_mask = cv2.erode(mask, erode)

        color_mask = cv2.dilate(mask, dilate)
        color_mask = cv2.dilate(mask, dilate)

        color_mask = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.displayImage(color_mask, 2)

    def update_frame_img(self):
        self.displayImage(self.image, 1)
        # Reference
        # lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} #assign new item lower['blue'] = (93, 10, 0)
        # upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}

        blurred = cv2.blur(self.image, (7, 7))
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        color_lower = np.array([self.h_min.value(), self.s_min.value(), self.v_min.value()], np.uint8)
        color_upper = np.array([self.h_max.value(), self.s_max.value(), self.v_max.value()], np.uint8)

        self.current_value.setText('Current Value -> Min :' + str(color_lower) + ' Max: ' + str(color_upper))

        mask = cv2.inRange(hsv, color_lower, color_upper)
        dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))
        erode = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

        color_mask = cv2.erode(mask, erode)
        color_mask = cv2.erode(mask, erode)

        color_mask = cv2.dilate(mask, dilate)
        color_mask = cv2.dilate(mask, dilate)

        color_mask = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.displayImage(color_mask, 2)

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR to RGB
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.image_label1.setPixmap(QPixmap.fromImage(outImage))
            self.image_label1.setScaledContents(True)

        if window == 2:
            self.image_label2.setPixmap(QPixmap.fromImage(outImage))
            self.image_label2.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorDetector()
    window.setWindowTitle('OpenCV Color Detector')
    window.show()
    sys.exit(app.exec_())