import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QDesktopWidget, QWidget,
                             QLabel, QLineEdit, QAction, QFileDialog)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # -----Widgets-----
        load_button = QPushButton("Load")
        pred_button = QPushButton("Identify")
        copy_button = QPushButton("Copy")
        display_bar = QLineEdit(self)
        display_img = QLabel()

        # -----Adjust-----

        # -----Layout-----
        main_layout = QVBoxLayout()
        main_layout.addWidget(display_img)

        # Operation bar: horizontal layout
        display_line = QHBoxLayout()
        display_line.addWidget(load_button, 1)
        display_line.addWidget(display_bar, 5)
        display_line.addWidget(pred_button, 1)
        display_line.addWidget(copy_button, 1)
        main_layout.addLayout(display_line)

        # -----Action-----
        # Open file action
        load_button.clicked.connect(self.show_openfile_dialog)

        # display_bar.textChanged[str].connect(self) # self.method

        # -----Windows-----
        self.setLayout(main_layout)
        self.resize(1000, 618)
        self.center()
        self.setWindowTitle("Bankcard Recognizer")
        self.show()

    def center(self):
        # Appear in the center of screen.
        window = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        window.moveCenter(center_point)
        self.move(window.topLeft())

    def show_openfile_dialog(self):
        # Action for showing openfile dialog
        fname = QFileDialog.getOpenFileName(self, caption="Open file", directory="c:", filter="*.png *.jpg *.jpeg")

        if fname[0]:
            with open(fname[0], 'r') as f:
                img = cv2.imread(f)
                pix = QPixmap(img)

                # self.widget.setXXX(img)

