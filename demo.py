# BankCard-Recognizer with GUI
#
# Author: Shawn Hu    © Copyright 2019
# License: MIT
#
# Usage: Run this demo with Python.
#
# Requirements: Numpy, Scipy, OpenCV2, Cython, Keras, TensorFlow-GPU.

import sys
from PyQt5.QtWidgets import QApplication
from gui.app import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()
    sys.exit(app.exec_())
