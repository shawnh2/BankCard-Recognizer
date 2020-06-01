# BankCard-Recognizer with GUI
#
# Author: Shawn Hu    © Copyright 2019
# License: MIT
#
# Usage: Run this demo with Python.
#

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui.main import UIMainWindow
from gui.app import APP


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_ui = UIMainWindow()
    main_app = APP(main_window)
    main_window.show()
    sys.exit(app.exec_())
