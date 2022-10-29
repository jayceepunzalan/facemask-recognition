# importing libraries
import sys
import cv2
import re
import os
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from PyQt5 import QtGui, QtCore

from os.path import expanduser
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    """alist.sort(key=natural_keys) sorts in human order"""
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        # Window properties
        self.setWindowTitle('Face Mask Detector Tool')
        self.setFixedSize(765, 500)

        # Browse folder directory
        self.image_dir = QLineEdit(self)
        self.image_dir.setPlaceholderText('Enter image directory')
        self.image_dir.resize(550, 27)
        self.image_dir.move(55, 150)
        self.image_dir.setStyleSheet("background-color: white;")
        self.image_dir.setReadOnly(True)
        self.image_dir.textChanged.connect(self.on_input)

        self.browseImageFolder_btn = QPushButton(self)
        self.browseImageFolder_btn.setText('Browse')
        self.browseImageFolder_btn.move(615, 150)   
        self.browseImageFolder_btn.clicked.connect(self.get_image_directory)

        self.startProcess_btn = QPushButton(self)
        self.startProcess_btn.setText('Start Process')
        self.startProcess_btn.move(330,250)
        self.startProcess_btn.setEnabled(False)
        self.startProcess_btn.clicked.connect(self.detect_facemask)

    # Choose image directory
    def get_image_directory(self):
        global filename
        image_directory = QFileDialog.getExistingDirectory(None, '[Images] Select a folder:', expanduser("~"))
        self.image_dir.setText(image_directory)
        filename = image_directory.split('/')[-1]
        if image_directory != "":
            num_files = len(os.listdir(image_directory))
            if num_files == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle('Warning!')
                msg.setText('Folder empty.\nFolder must contain image file.')
                msg.exec_()
                self.image_dir.setText('')
                
            else:
                for x in os.listdir(image_directory):
                    x = x.upper()
                    if x.endswith('.JPG') or x.endswith('.JPEG') or x.endswith('.PNG'):
                        True
                    else:       
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setWindowTitle('Warning!')
                        msg.setText('Folder must contain image file only.')
                        msg.exec_()
                        self.image_dir.setText('')
                        break

    # Show Results Window
    def show_results_window(self):
        Result_Window()


    def detect_facemask(self, image_directory, is_image=True):
        self.behaviour_during_detection()
        image_directory = self.image_dir.text()

        folder_files = os.listdir(image_directory)
        folder_files.sort(key=natural_keys)
        
        model = load_model("model_best_weights.h5")

        for filename in folder_files:
            # print()
            # print(f'Image name: {filename}')

            original_face = cv2.imread(os.path.join(image_directory, filename))
            face = cv2.cvtColor(original_face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (200, 200))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            face_list = []
            face_list.append(face)

            faces = np.array(face_list, dtype="float32")
            preds = model.predict(faces, batch_size=32)

            label = np.argmax(preds)
            label = 'Mask' if label==0 else 'Unmask'

            confidence = round(max(preds[0])*100, 2)
            original_face = cv2.resize(original_face, (800, 600))
            cv2.putText(original_face, f'Tag: {label}',(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
            cv2.imshow(f"Facemask Detection Tool - {filename}", original_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def behaviour_during_detection(self):
        self.browseImageFolder_btn.setEnabled(False)
        self.startProcess_btn.setEnabled(False)


    @QtCore.pyqtSlot()
    def on_input(self):
        self.startProcess_btn.setEnabled(
        (bool(self.image_dir.text()))
        )



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
