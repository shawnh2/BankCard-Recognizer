import os
import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QDesktopWidget, QWidget,
                             QLabel, QLineEdit, QTextEdit, QFileDialog)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.prefix = "gui/icon/"
        self.last_path = "c:"
        self.warn_text = '<font color="red">{}</font>'
        self.tips_text = '<font color="green">{}</font>'
        self.alert = '<font color="orange">{}</font>'

        # -----Widgets-----
        self.load_button = QPushButton("Load")
        self.pred_button = QPushButton("Identify")
        self.copy_button = QPushButton("Copy")
        self.display_bar = QLineEdit(self)
        self.display_img = QLabel()
        self.logging_bar = QTextEdit("Logging begin at {}".format(datetime.datetime.now()))

        # -----Adjust-----
        self.display_img.setFixedSize(970, 550)
        self.display_img.setStyleSheet("QLabel{background:white}")
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
        self.setFixedSize(1280, 618)
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
            self.logging_bar.append("[*] Choose an image from %s" % name)
            img = QPixmap(name).scaled(self.display_img.width(), self.display_img.height())
            self.display_img.setPixmap(img)
            self.logging_bar.append(self.tips_text.format("[*] Display it successfully."))
        else:
            self.logging_bar.append("[*] Cancel this loading process.")

    def copy_to_clipboard(self):
        # Action for copying the identify result to clipboard
        text = self.display_bar.text()
        try:
            import clipboard

            clipboard.copy(text)
            self.copy_button.setText("OK")
            self.logging_bar.append(self.tips_text.format("[*] Copy to clipboard successfully."))
        except ImportError:
            self.copy_button.setText("Fail")
            self.logging_bar.append(self.warn_text.format("[!] Failure happened: missing component."))
            self.logging_bar.append(self.warn_text.format("[!] Try install this component by typing:"))
            self.logging_bar.append(self.alert.format(" 'pip install clipboard' "))
            self.logging_bar.append(self.warn_text.format("[!] Or copy it manually."))

    def predict(self):
        # Action for identify the bankcard
        try:
            pass
        except ImportError as e:
            self.logging_bar.append(self.warn_text.format("[!] Missing important module."))
            self.logging_bar.append(self.warn_text.format("[!] ImportError:" + str(e)))
            self.logging_bar.append(self.warn_text.format("[!] Please install requirement module and retry it later."))
