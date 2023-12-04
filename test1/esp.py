import sys
import time

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def set_rect(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor("#00FF00"))
        painter.setPen(pen)
        painter.drawRect(self.x, self.y, self.w, self.h)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 创建 OverlayWindow 对象
    overlay_window = OverlayWindow()

    def update_rect(x, y, w, h):
        overlay_window.set_rect(x, y, w, h)
        overlay_window.show()

    # 示例代码：传入动态的矩形参数
    x = 50
    y = 50
    w = 200
    h = 200
    # 调用 update_rect 函数更新矩形显示
    update_rect(x, y, w, h)


    sys.exit(app.exec_())