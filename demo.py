# BankCard-Recognizer with GUI
#
# Author: Shawn Hu    © Copyright 2019
# License: MIT
#
# Usage: Run this demo with Python.
#
# Requirements: Numpy, Scipy, OpenCV2, Cython, Keras, TensorFlow-GPU,
#               PyQt5, clipboard.

import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui.main import UIMainWindow
from gui.app import APP

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_ui = UIMainWindow()
    main_app = APP(main_window)
    main_window.show()
    sys.exit(app.exec_())
