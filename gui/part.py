from PyQt5.QtGui import QPen, QCursor
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem


class DisplayView(QGraphicsView):

    def __init__(self, parent=None):
        super(DisplayView, self).__init__(parent)

    def set_item(self, img_item):
        self.item = DisplayPixmapItem(img_item)
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

    def __init__(self, pix, parent=None):
        super(DisplayPixmapItem, self).__init__(parent)
        self.setPixmap(pix)
        # Some useful attributes
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

    def paint(self, painter, QStyleOptionGraphicsItem, QWidget):
        super(DisplayPixmapItem, self).paint(painter, QStyleOptionGraphicsItem, QWidget)
        if self.activate:
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            if self.current_point is None:
                return
            painter.drawRect(QRectF(self.start_point, self.current_point))
            self.update()
