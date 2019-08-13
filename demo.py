# BankCard-Recognizer with GUI
#
# Author: Shawn Hu    © Copyright 2019
# License: MIT
#
# Usage: Run this demo with Python.
#
# Requirements: Numpy, Scipy, OpenCV2, Cython, Keras, TensorFlow-GPU,
#               PyQt5, clipboard.

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui.main import UIMainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = UIMainWindow()
    ui.setup_ui(main_window)
    main_window.show()
    sys.exit(app.exec_())
