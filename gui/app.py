import os
import datetime

import cv2
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPainter, QPen, QImage
from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QDesktopWidget, QWidget,
                             QLabel, QLineEdit, QTextEdit, QFileDialog)
from gui.utils import selected_box, log_text


class DisplayView(QLabel):
    """
    A custom Label, so can load image and display it.
    With several override methods, now is able to draw a
    rectangle on the image, so will be convenient to locate
    the number of bankcard.
    """

    def __init__(self, logging, led):
        super().__init__()
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.flag = False
        self.logging = logging
        self.img_path = None
        self.led = led

    def mousePressEvent(self, QMouseEvent):
        self.flag = True
        self.x0 = QMouseEvent.x()
        self.y0 = QMouseEvent.y()
        self.logging.append("[*] Draw a rectangle start from:")
        self.logging.append("    (x0, y0) = ({}, {})".format(self.x0, self.y0))

    def mouseReleaseEvent(self, QMouseEvent):
        self.flag = False
        self.logging.append("[*] Draw a rectangle stop from:")
        self.logging.append("    (x1, y1) = ({}, {})".format(self.x1, self.y1))
        # Get selected area of image to display
        if self.img_path is None:
            self.logging.append(log_text("[*] Invalid selection! It's NULL!", "warning"))
        else:
            img_array = cv2.imread(self.img_path)
            selected_area = selected_box(img_array, self.x0, self.y0, self.x1, self.y1)
            cv2.imshow("test", selected_area)

    def mouseMoveEvent(self, QMouseEvent):
        if self.flag:
            self.x1 = QMouseEvent.x()
            self.y1 = QMouseEvent.y()
            self.update()

    def paintEvent(self, QPaintEvent):
        super().paintEvent(QPaintEvent)
        rect = QRect(self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(rect)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.prefix = "gui/icon/"
        self.last_path = "./dataset/test/"

        # -----Widgets-----
        self.load_button = QPushButton("Load")
        self.pred_button = QPushButton("Identify")
        self.copy_button = QPushButton("Copy")
        self.display_bar = QLineEdit(self)
        self.logging_bar = QTextEdit("Logging begin at {}".format(datetime.datetime.now()))
        self.select_area = QLabel()
        self.display_img = DisplayView(self.logging_bar, self.select_area)

        # -----Adjust-----
        self.display_img.setFixedSize(970, 550)
        self.display_img.setStyleSheet("QLabel{background:white}")
        self.display_img.setToolTip("Displaying area.")
        self.select_area.setFixedSize(970, 95)
        self.select_area.setStyleSheet("QLabel{background:gray}")
        self.select_area.setToolTip("Selecting area.")
        self.display_bar.setFixedHeight(30)
        self.display_bar.setFont(QFont("Timers", 15))

        self.load_button.setFixedHeight(30)
        self.load_button.setIcon(QIcon(self.prefix + "load.png"))
        self.load_button.setToolTip("Load in an image about bankcard.")
        self.pred_button.setFixedHeight(30)
        self.pred_button.setIcon(QIcon(self.prefix + "predict.png"))
        self.pred_button.setToolTip("Start to identify the bankcard number.")
        self.copy_button.setFixedHeight(30)
        self.copy_button.setIcon(QIcon(self.prefix + "copy.png"))
        self.copy_button.setToolTip("Copy to your clipboard.")

        self.logging_bar.setStyleSheet("QTextEdit{background: rgb(240, 240, 240)}")
        self.logging_bar.setReadOnly(True)
        self.logging_bar.append("BankCardRecognizer by ShawnHu 2019")

        # -----Layout-----
        # Left - Row 1
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.display_img, 1, Qt.AlignCenter)
        # Left - Row 2
        main_layout.addWidget(self.select_area, 1, Qt.AlignCenter)
        # Left - Row 3
        display_line = QHBoxLayout()
        display_line.addWidget(self.load_button, 1)
        display_line.addWidget(self.display_bar, 5)
        display_line.addWidget(self.pred_button, 1)
        display_line.addWidget(self.copy_button, 1)
        main_layout.addLayout(display_line)
        # Right - Row 1
        total_layout = QHBoxLayout()
        total_layout.addLayout(main_layout)
        total_layout.addWidget(self.logging_bar, 1)
        total_layout.setSpacing(10)

        # -----Action-----
        self.load_button.clicked.connect(self.show_openfile_dialog)
        self.pred_button.clicked.connect(self.predict)
        self.copy_button.clicked.connect(self.copy_to_clipboard)

        # -----Windows-----
        self.setLayout(total_layout)
        self.setFixedSize(1280, 720)
        self.setWindowTitle("Bankcard Recognizer")
        self.setWindowIcon(QIcon(self.prefix + "bankcard.png"))
        self.center()
        self.show()

    def center(self):
        # Appear in the center of screen.
        window = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        window.moveCenter(center_point)
        self.move(window.topLeft())

    def show_openfile_dialog(self):
        # Action for showing openfile dialog
        name, type = QFileDialog.getOpenFileName(self,
                                                 caption="Open file",
                                                 directory=self.last_path,
                                                 filter="Image(*.png *.jpg *.jpeg)")
        if name:
            self.last_path = os.path.split(name)[0]
            self.display_img.img_path = name
            self.logging_bar.append("[*] Choose an image from %s" % name)
            pix_img = QPixmap(name).scaled(self.display_img.width(), self.display_img.height())
            self.display_img.setPixmap(pix_img)
            self.display_img.setCursor(Qt.CrossCursor)
            self.logging_bar.append(log_text("[*] Display it successfully.", "ok"))
        else:
            self.logging_bar.append("[*] Cancel this loading process.")

    def copy_to_clipboard(self):
        # Action for copying the identify result to clipboard
        text = self.display_bar.text()
        try:
            import clipboard

            clipboard.copy(text)
            self.copy_button.setText("OK")
            self.logging_bar.append(log_text("[*] Copy to clipboard successfully.", "ok"))
        except ImportError:
            self.copy_button.setText("Fail")
            self.logging_bar.append(log_text("[!] Failure happened: missing component.", "error"))
            self.logging_bar.append(log_text("[!] Try install this component by typing:", "error"))
            self.logging_bar.append(log_text(" 'pip install clipboard' ", "warning"))
            self.logging_bar.append(log_text("[!] Or copy it manually.", "error"))

    def predict(self):
        # Action for identify the bankcard
        try:
            pass
        except ImportError as e:
            self.logging_bar.append(log_text("[!] Missing important module.", "error"))
            self.logging_bar.append(log_text("[!] ImportError:" + str(e), "error"))
            self.logging_bar.append(log_text("[!] Please install requirement module and retry it later.", "error"))
