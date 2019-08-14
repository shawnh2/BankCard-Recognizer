import os

import cv2
import cgitb
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from gui.main import UIMainWindow
from gui.utils import rotate_bound, array_to_pixmap

# Prevent closing and show its error in stdout.
cgitb.enable(format("text"))


class APP(UIMainWindow):
    """Load in all kinds of actions from UIMainWindow."""

    def __init__(self, q_main_window):
        super().setup_ui(q_main_window)
        self.last_path = "./dataset/test/"
        self.zoom_scala = 1
        self.rotate_degree = 0
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
            self.img_array = cv2.imread(name)
            self.pose_it(self.img_array.copy())
        else:
            pass

    def pose_it(self, img_array):
        pix = array_to_pixmap(img_array)
        self.diaplay_img.set_item_with_stage(pix, img_array, self.selected_img)
        self.diaplay_img.set_scene()

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

    def click_rotate_right(self):
        # Negative number means right.
        if self.diaplay_img.scene():
            self.rotate_degree -= 1
            if self.rotate_degree <= -3:
                self.rotate_degree = 0
            self.img_array = rotate_bound(self.img_array, self.rotate_degree)
            self.diaplay_img.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
            self.pose_it(self.img_array.copy())
        else:
            self.statusbar.showMessage("No image load in yet!")

    def click_rotate_left(self):
        if self.diaplay_img.scene():
            self.rotate_degree += 1
            if self.rotate_degree >= 3:
                self.rotate_degree = 0
            self.img_array = rotate_bound(self.img_array, self.rotate_degree)
            self.diaplay_img.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
            self.pose_it(self.img_array.copy())
        else:
            self.statusbar.showMessage("No image load in yet!")
