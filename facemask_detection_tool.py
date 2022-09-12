# importing libraries
import sys
import os

from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from PyQt5 import QtGui, QtCore

class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        # Window properties
        self.setWindowTitle('Face Mask Detector Tool')
        self.setFixedSize(765, 500)

        # changing the background color to gray
        self.setStyleSheet("background-color: grey;")

        # Browse folder directory
        self.image_dir = QLineEdit(self)
        self.image_dir.setPlaceholderText('Enter image directory')
        self.image_dir.resize(550, 27)
        self.image_dir.move(55, 150)
        self.image_dir.setStyleSheet("background-color: white;")
        self.image_dir.setReadOnly(True)

        self.browseImageFolder_btn = QPushButton(self)
        self.browseImageFolder_btn.setText('Browse')
        self.browseImageFolder_btn.move(615, 150)
        self.browseImageFolder_btn.setStyleSheet("background-color: violet;")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
