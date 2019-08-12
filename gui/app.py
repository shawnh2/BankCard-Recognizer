from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout, QDesktopWidget, QWidget,
                             QLabel, QLineEdit, QFileDialog)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.prefix = "gui/icon/"

        # -----Widgets-----
        self.load_button = QPushButton("Load")
        self.pred_button = QPushButton("Identify")
        self.copy_button = QPushButton("Copy")
        self.display_bar = QLineEdit(self)
        self.display_img = QLabel()

        # -----Adjust-----
        self.display_img.setFixedSize(970, 550)
        self.display_img.setStyleSheet("QLabel{background:white}")
        self.load_button.setFixedHeight(30)
        self.load_button.setIcon(QIcon(self.prefix + "load.png"))
        self.load_button.setToolTip("Load in an image about bankcard.")
        self.pred_button.setFixedHeight(30)
        self.pred_button.setIcon(QIcon(self.prefix + "predict.png"))
        self.pred_button.setToolTip("Start to identify the bankcard number.")
        self.copy_button.setFixedHeight(30)
        self.copy_button.setIcon(QIcon(self.prefix + "copy.png"))
        self.copy_button.setToolTip("Copy to your clipboard.")
        self.display_bar.setFixedHeight(30)

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
        self.load_button.clicked.connect(self.show_openfile_dialog)
        self.pred_button.clicked.connect(self.predict)
        self.copy_button.clicked.connect(self.copy_to_clipboard)

        # -----Windows-----
        self.setLayout(main_layout)
        self.setFixedSize(1000, 618)
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
                                                 directory="c:",
                                                 filter="Image(*.png *.jpg *.jpeg)")
        if name:
            img = QPixmap(name).scaled(self.display_img.width(), self.display_img.height())
            self.display_img.setPixmap(img)

    def copy_to_clipboard(self):
        # Action for copying the identify result to clipboard
        text = self.display_bar.text()
        try:
            import win32clipboard as win
            import win32con

            win.OpenClipboard()
            win.EmptyClipboard()
            win.SetClipboardData(win32con.CF_TEXT, text)
            win.CloseClipboard()
            self.copy_button.setText("OK")
            self.copy_button.setToolTip("Now is in your clipboard you can also continue.")
        except ImportError:
            self.copy_button.setText("Fail")
            self.copy_button.setToolTip("Failure reason: missing win32con in python.")

    def predict(self):
        # Action for identify the bankcard
        pass
