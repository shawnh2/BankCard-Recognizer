from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QDesktopWidget, QWidget,
                             QLabel, QLineEdit, QFileDialog)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        # -----Widgets-----
        self.load_button = QPushButton("Load")
        self.pred_button = QPushButton("Identify")
        self.copy_button = QPushButton("Copy")
        self.display_bar = QLineEdit(self)
        self.display_img = QLabel()

        # -----Adjust-----
        self.display_img.setFixedSize(970, 550)
        self.display_img.setStyleSheet("QLabel{background:white}")

        # -----Layout-----
        # Row 1
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.display_img, 1, Qt.AlignCenter)
        # Row 2
        display_line = QHBoxLayout()
        display_line.addWidget(self.load_button, 1)
        display_line.addWidget(self.display_bar, 5)
        display_line.addWidget(self.pred_button, 1)
        display_line.addWidget(self.copy_button, 1)
        main_layout.addLayout(display_line)

        # -----Action-----
        # Open file action
        self.load_button.clicked.connect(self.show_openfile_dialog)

        # -----Windows-----
        self.setLayout(main_layout)
        self.setFixedSize(1000, 618)
        self.setWindowTitle("Bankcard Recognizer")
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
                                                 directory="c:",
                                                 filter="Image(*.png *.jpg *.jpeg)")
        img = QPixmap(name).scaled(self.display_img.width(), self.display_img.height())
        self.display_img.setPixmap(img)

    def predict(self):
        # Action for identify the bankcard
        pass

    def copy_to_clipboard(self):
        # Action for copying the identify result to clipboard
        pass
