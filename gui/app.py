import os

import cv2
import cgitb
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QMessageBox

from east.predict import predict_txt
from crnn.predict import single_recognition
from gui.main import UIMainWindow
from gui.utils import rotate_bound, array_to_pixmap, hard_coords, selected_box

# Prevent closing and show its error in stdout.
cgitb.enable(format("text"))


class APP(UIMainWindow):
    """Load in all kinds of actions from UIMainWindow."""

    def __init__(self, q_main_window):
        super().setup_ui(q_main_window)
        self.last_path = "./dataset/test/"
        self.model_path = ["./crnn/model/crnn_model.h5",
                           "./east/model/east_model_weights.h5"]
        self.zoom_scala = 1
        self.rotate_degree = 0
        self.img_name = None
        # Activate actions
        self.functions()

    def functions(self):
        self.load_button.clicked.connect(self.load_img_from_filedialog)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.zoom_in.clicked.connect(self.click_zoom_in)
        self.zoom_out.clicked.connect(self.click_zoom_out)
        self.rotate_left.clicked.connect(self.click_rotate_left)
        self.rotate_right.clicked.connect(self.click_rotate_right)
        self.identify_button.clicked.connect(self.predict)

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
        self.img_name, _ = QFileDialog.getOpenFileName(None, "Load Image", self.last_path, "*.png;*.jpg;*.jpeg")
        self.diaplay_img.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
        if self.img_name:
            # Saving current path for next load convenience.
            self.last_path = os.path.split(self.img_name)[0]
            # Refresh it every time new image load in.
            self.zoom_scala = 1
            self.rotate_degree = 0
            # Load in new image and its info.
            self.img_array = cv2.imread(self.img_name)
            self.pose_it(self.img_array.copy())

    def load_model_from_filedialog(self, type: int):
        # 0 is crnn model, 1 is east model
        name, ext = QFileDialog.getOpenFileName(None, "Load Model", self.last_path, "*.h5")
        if name:
            self.model_path[type] = name

    def pose_it(self, img_array):
        # It will not change original img_array, just a copy.
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

    def fail_msg(self):
        # While auto location sometimes failed.
        QMessageBox.warning(None,
                            "Error!",
                            "Auto location failed! Please try manual.",
                            QMessageBox.Yes)

    def predict(self):
        # Check model.
        if not self.diaplay_img.scene():
            self.statusbar.showMessage("Please load in an Image then start.")
            return
        if not os.path.exists(self.model_path[0]):
            self.load_model_from_filedialog(0)
        if not os.path.exists(self.model_path[1]):
            self.load_model_from_filedialog(1)

        if self.diaplay_img.item.activate:
            # Manual-Locate
            pred_img = self.diaplay_img.item.selected_img
            self.predict_selected_img(pred_img)
        else:
            # Auto-Locate
            result = predict_txt(self.img_name, self.model_path[1])
            result_array = cv2.imread(self.img_name)
            if len(result):
                result = hard_coords(result[0])
                pred_img = selected_box(result_array, *result)
                self.set_selected_img_stage(pred_img)
                self.predict_selected_img(pred_img)
            else:
                self.fail_msg()

    def predict_selected_img(self, pred_img):
        if pred_img is None:
            self.statusbar.showMessage("Please select an area then start identify.")
            return
        result = single_recognition(pred_img, (256, 32), self.model_path[0])
        self.result_line.setText(result)

    def set_selected_img_stage(self, pred_img):
        selected_pix = QGraphicsPixmapItem(array_to_pixmap(pred_img))
        selected_scene = QGraphicsScene()
        selected_scene.addItem(selected_pix)
        self.selected_img.setScene(selected_scene)
        self.selected_img.fitInView(selected_pix)
