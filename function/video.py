from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, QObject
class ImageLabel(QLabel):
    clicked = pyqtSignal(QObject)

    def __init__(self,image, parent=None):
        super().__init__(parent)
        self.setPixmap(QPixmap(image))

    def mousePressEvent(self, event):
        self.clicked.emit(self)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setting = QtWidgets.QWidget()  # 创建窗口主部件

        grid = QGridLayout(self)

        self.label1 = ImageLabel("data/gesture/gesture3.jpg")
        self.label1.clicked.connect(self.on_label_clicked)
        grid.addWidget(self.label1, 0, 0)
        self.label2 = ImageLabel("data/gesture/gesture3.jpg")
        self.label2.clicked.connect(self.on_label_clicked)
        grid.addWidget(self.label2, 0, 1)
        self.label3 = ImageLabel("data/gesture/gesture3.jpg")
        self.label3.clicked.connect(self.on_label_clicked)
        grid.addWidget(self.label3, 0, 2)
        self.setting .show()
    def on_label_clicked(self, sender):
        if sender == self.label1:
            print('Label 1 clicked')
        elif sender == self.label2:
            print('Label 2 clicked')
        elif sender == self.label3:
            print('Label 3 clicked')

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
