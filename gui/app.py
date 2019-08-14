import os

import cv2
import cgitb
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem

from gui.main import UIMainWindow

# Prevent closing and show its error in stdout.
cgitb.enable(format("text"))


class APP(UIMainWindow):
    """Load in all kinds of actions from UIMainWindow."""

    def __init__(self, q_main_window):
        super().setup_ui(q_main_window)
        self.last_path = "./dataset/test/"
        self.zoom_scala = 1
        self.rotate_degree = 0
        self.rotate_center_point = ()  # (x, y)
        # Activate actions
        self.functions()

    def functions(self):
        self.load_button.clicked.connect(self.load_img_from_filedialog)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.zoom_in.clicked.connect(self.click_zoom_in)
        self.zoom_out.clicked.connect(self.click_zoom_out)
        self.rotate_left.clicked.connect(self.click_rotate_left)
        self.rotate_right.clicked.connect(self.click_rotate_right)

    def copy_to_clipboard(self):
        # Action for copying the identify result to clipboard.
        text = self.result_line.text()
        try:
            import clipboard
            clipboard.copy(text)
            self.statusbar.showMessage(" Copy successfully!")
        except ImportError:
            self.statusbar.showMessage(" Copy failed because of missing module: clipboard."
                                       " Try '$ pip install clipboard' to install it.")

    def load_img_from_filedialog(self):
        name, ext = QFileDialog.getOpenFileName(None, "Load Image", self.last_path, "*.png;*.jpg;*.jpeg")
        if name:
            # Saving current path for next load convenience.
            self.last_path = os.path.split(name)[0]
            # Refresh it every time new image load in.
            self.zoom_scala = 1
            self.rotate_degree = 0
            # Load in new image and its info.
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            y, x, _ = img.shape
            bytes_per_line = 3 * x
            self.rotate_center_point = (x/2, y/2)
            # Add it to Widget.
            frame = QImage(img.data, x, y, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.diaplay_img.set_item(pix)
            self.diaplay_img.set_scene()
            # Adjust some attributes.
            self.diaplay_img.item.setTransformOriginPoint(*self.rotate_center_point)
        else:
            pass

    def click_zoom_in(self):
        if self.diaplay_img.scene():
            self.zoom_scala += 0.05
            # Add a threshold.
            if self.zoom_scala >= 1.5:
                self.zoom_scala = 1.5
            self.diaplay_img.item.setScale(self.zoom_scala)
        else:
            self.statusbar.showMessage("No image load in yet!")

    def click_zoom_out(self):
        if self.diaplay_img.scene():
            self.zoom_scala -= 0.05
            if self.zoom_scala <= 0.1:
                self.zoom_scala = 0.1
            self.diaplay_img.item.setScale(self.zoom_scala)
        else:
            self.statusbar.showMessage("No image load in yet!")

    def click_rotate_left(self):
        # Negative number means left.
        if self.diaplay_img.scene():
            self.rotate_degree -= 1
            if self.rotate_degree <= -90:
                self.rotate_degree = -90
            self.diaplay_img.item.setRotation(self.rotate_degree)
        else:
            self.statusbar.showMessage("No image load in yet!")

    def click_rotate_right(self):
        if self.diaplay_img.scene():
            self.rotate_degree += 1
            if self.rotate_degree >= 90:
                self.rotate_degree = 90
            self.diaplay_img.item.setRotation(self.rotate_degree)
        else:
            self.statusbar.showMessage("No image load in yet!")
