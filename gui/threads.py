import cv2
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox, QGraphicsPixmapItem, QGraphicsScene

from crnn.predict import single_recognition
from east.predict import predict_txt
from gui.utils import hard_coords, selected_box, array_to_pixmap


class PredictThread(QThread):

    def __init__(self, image, shape: tuple, model_path: str, container):
        super().__init__()
        self.image = image
        self.shape = shape
        self.model_path = model_path
        self.container = container

    def run(self):
        result = single_recognition(self.image, self.shape, self.model_path)
        self.container.setText(result)


class AutoLocateThread(QThread):

    def __init__(self, image_path: str, model_path: dict, img_container, txt_container):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.img_container = img_container
        self.txt_container = txt_container

    def run(self):
        result = predict_txt(self.image_path, self.model_path["east"])
        result_array = cv2.imread(self.image_path)
        if len(result):
            result = hard_coords(result[0])
            predict = selected_box(result_array, *result)
            # Stage image
            selected_pix = QGraphicsPixmapItem(array_to_pixmap(predict))
            selected_scene = QGraphicsScene()
            selected_scene.addItem(selected_pix)
            self.img_container.setScene(selected_scene)
            self.img_container.fitInView(selected_pix)
            # Predict number
            thread = PredictThread(predict, (256, 32), self.model_path["crnn"], self.txt_container)
            thread.run()
        else:
            self.fail_msg()

    def fail_msg(self):
        # While auto location sometimes failed.
        QMessageBox.warning(None,
                            "Error!",
                            "Auto location failed! Please try manual.",
                            QMessageBox.Yes)
