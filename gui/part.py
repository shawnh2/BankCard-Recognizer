from PyQt5.QtGui import QPen, QCursor
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem

from gui.utils import selected_box, array_to_pixmap


class DisplayView(QGraphicsView):

    def __init__(self, parent=None):
        super(DisplayView, self).__init__(parent)

    def set_item_with_stage(self, img_item, img_array, stage):
        self.item = DisplayPixmapItem(img_item, img_array, stage)
        # set this so mouse release event can listen.
        self.item.setFlag(QGraphicsItem.ItemIsMovable)

    def set_scene(self):
        # Prevent having trouble with method: scene()
        self.scene_ = QGraphicsScene()
        self.scene_.addItem(self.item)
        self.setScene(self.scene_)

    def mouseDoubleClickEvent(self, event):
        # Activate manual locate.
        if self.scene() is not None:
            if not self.item.activate:
                self.item.activate = True
                self.viewport().setProperty("cursor", QCursor(Qt.CrossCursor))
            else:
                self.item.activate = False
                self.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
        else:
            pass


class DisplayPixmapItem(QGraphicsPixmapItem):

    def __init__(self, pix, origin_img_array, stage, parent=None):
        super(DisplayPixmapItem, self).__init__(parent)
        self.setPixmap(pix)
        self.img_array = origin_img_array
        self.stage = stage
        # Some useful attributes
        self.selected_img = None
        self.start_point = None
        self.current_point = None
        self.activate = False
        self.flag = False

    def mousePressEvent(self, event):
        super(DisplayPixmapItem, self).mousePressEvent(event)
        if self.activate and self.scene() is not None:
            self.flag = True
            self.start_point = event.pos()
            self.current_point = None
        else:
            pass

    def mouseMoveEvent(self, event):
        if self.flag:
            self.current_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.activate:
            self.flag = False
            # Manual locate and display it in select area.
            try:
                x0, y0 = int(self.start_point.x()), int(self.start_point.y())
                x1, y1 = int(self.current_point.x()), int(self.current_point.y())
                self.selected_img = selected_box(self.img_array, x0, y0, x1, y1)
                selected_pix = QGraphicsPixmapItem(array_to_pixmap(self.selected_img))
                selected_scene = QGraphicsScene()
                selected_scene.addItem(selected_pix)
                self.stage.setScene(selected_scene)
                self.stage.fitInView(selected_pix)
            except AttributeError:
                self.selected_img = None
                return

    def paint(self, painter, QStyleOptionGraphicsItem, QWidget):
        super(DisplayPixmapItem, self).paint(painter, QStyleOptionGraphicsItem, QWidget)
        if self.activate:
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            if self.current_point is None:
                return
            painter.drawRect(QRectF(self.start_point, self.current_point))
            self.update()
