import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pickle
import os
from pyqtgraph import ImageView, RectROI
from tqdm import tqdm
import cv2


class ImageCropper(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.n = 360
        self.w = 600
        self.c = 0
        self.timer = QtCore.QTimer(self, timeout=self.update)
        self.is_paused = True
        self.initUI()

    def initUI(self):
        self.setWindowTitle('StringArt')
        self.roi = None
        self.item = None

        self.imageView = ImageView(self)
        self.setCentralWidget(self.imageView)

        openFile = QtWidgets.QAction('Open', self)
        openFile.triggered.connect(self.openImage)
        saveSteps = QtWidgets.QAction('Save steps', self)
        saveSteps.triggered.connect(self.saveSteps)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveSteps)

        self.imageView.ui.histogram.hide()
        self.imageView.ui.roiBtn.hide()
        self.imageView.ui.menuBtn.hide()

        self.setGeometry(100, 100, 800, 600)
        self.show()

    def openImage(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.webp)", options=options)
        if fileName:
            self.loadImage(fileName)

    def saveSteps(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save steps File", "StringStep.txt", "Text File (*.txt)", options=options)
        if not fileName:
            return
        if not fileName.endswith('.txt'):
            fileName += '.txt'
        with open(fileName, 'w') as f:
            f.write(f"# {self.n} nails\n# Total {len(self.step)-1} steps\n\n")
            for i, nail in enumerate(self.step):
                f.write(f"{i}\t{nail}\n")

    def loadImage(self, fileName):
        self.image = QtGui.QImage(fileName)
        self.imageArray = self.qimageToNumpyArray(self.image)[..., ::-1].transpose(1, 0, 2)
        self.imageView.setImage(self.imageArray)
        if self.roi is None:
            self.createROI()
            self.roi.sigRegionChangeFinished.connect(self.cropImage)
        self.imageView.addItem(self.roi)

    def createROI(self):
        self.roi = RectROI([20, 20], [100, 100], pen='r', resizable=False, aspectLocked=True)

    def qimageToNumpyArray(self, image):
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        arr = arr[:, :, :3]
        return np.require(arr, np.uint8, 'C')

    def cropImage(self):
        if hasattr(self, 'roi') and hasattr(self, 'imageArray'):
            y1, x1 = self.roi.pos()
            y2, x2 = self.roi.size()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            self.center = [x1 + x2 / 2, y1 + y2 / 2]
            self.size_ = x2 / 2
            img = self.imageArray[y1:y1 + x2, x1:x1 + y2]
            w = self.w
            cropped_img = cv2.resize(img, (w, w))
            mask = ((np.mgrid[-1:1:w * 1j, -1:1:w * 1j] ** 2).sum(0) ** .5) > 1
            cropped_img[mask] = 0
            self.gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY).astype(float)
            self.calcEdges()
            self.string_art()

    def calcEdges(self):
        n = self.n
        w = self.w
        file_name = f'StringArtEdges{n}_{w}.pkl'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                self.con = pickle.load(f)
        else:
            cos, sin = np.cos(np.r_[:2 * np.pi:2 * np.pi / n]), np.sin(np.r_[:2 * np.pi:2 * np.pi / n])
            lin = np.c_[:1:2j * w]
            self.con = []
            for i in tqdm(range(n)):
                a = (((cos - cos[i]) * lin + cos[i] + 1) / 2 * w).astype(int).clip(0, w - 1).tolist()
                b = (((sin - sin[i]) * lin + sin[i] + 1) / 2 * w).astype(int).clip(0, w - 1).tolist()
                c = np.dstack([a, b])
                self.con.append({j: np.unique(k, axis=1) for j, k in enumerate(c.transpose(1, 2, 0)) if i != j})
            with open(file_name, 'wb') as f:
                pickle.dump(self.con, f)

    def string_art(self):
        n = 360
        c = self.c
        self.temperature = 30
        self.nail = np.random.randint(0, n)
        self.step = [self.nail]
        self.eim = self.gray_img.astype(float)
        view = self.imageView
        self.pos = np.c_[np.cos(np.r_[:2 * np.pi:2 * np.pi / n]) * self.size_ - self.size_ - 20,
                         np.sin(np.r_[:2 * np.pi:2 * np.pi / n]) * self.size_ + self.center[0]]
        self.adj = np.array([], int).reshape(0, 2)
        if self.item is None:
            self.item = pg.GraphItem()
            view.addItem(self.item)
        self.item.setData(pos=self.pos, adj=self.adj, pen=pg.mkPen(['w', 'k'][c], width=.2), symbolBrush=['w', 'k'][c])

    def update(self):
        c = self.c
        _nail = (max, min)[c](self.con[self.nail],
                               key=lambda x: self.eim[self.con[self.nail][x][0], self.con[self.nail][x][1]].mean())
        a, b = self.con[self.nail][_nail]
        self.eim[a, b] -= self.temperature * (-1) ** c
        self.adj = np.vstack([self.adj, [self.nail, _nail]])
        self.item.setData(pos=self.pos, adj=self.adj, pen=pg.mkPen(['w', 'k'][c], width=.2),
                          symbolBrush=['w', 'k'][c])
        self.nail = _nail
        self.step.append(self.nail)

    def toggle_pause(self):
        if self.is_paused:
            self.timer.start(1)
        else:
            self.timer.stop()
        self.is_paused = not self.is_paused

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.toggle_pause()
        elif event.key() == QtCore.Qt.Key_B:
            self.c = 1
            self.imageView.getView().setBackgroundColor('w')
            self.string_art()
        elif event.key() == QtCore.Qt.Key_W:
            self.c = 0
            self.imageView.getView().setBackgroundColor('k')
            self.string_art()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ImageCropper()
    sys.exit(app.exec_())
