import sys
from PyQt5.QtWidgets import QApplication
from gui.app import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()
    sys.exit(app.exec_())
