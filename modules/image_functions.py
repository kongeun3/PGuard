import os
import shutil

from PySide6.QtWidgets import QGraphicsScene, QFileDialog

from .utils import readImageAndPixmap


class ImageClass():
    def __init__(self, ui, main_window):
        # Inherit Class
        self.ui = ui
        self.main = main_window

        # Initialize Treeview
        self.main.fileModel.setRootPath(self.main.image_save_folder)
        self.ui.ImagetreeView.setModel(self.main.fileModel)
        self.ui.ImagetreeView.setRootIndex(self.main.fileModel.index(self.main.image_save_folder))

        # Function Connection
        self.ui.ImagetreeView.selectionModel().selectionChanged.connect(self.openImage)
        self.ui.addImageButton.clicked.connect(self.addImage)
        self.ui.deleteImageButton.clicked.connect(self.deleteImage)

        # Initialize Class
        print("load image class")
    
    def openImage(self, index):
        indexes = index.indexes() # QItemSelection에서 QModelIndex 리스트를 가져옴
        if indexes: # 선택된 항목이 하나 이상 있다면
            # 첫 번째로 선택된 항목의 파일 경로를 가져옴
            self.main.plot_image_path = self.main.fileModel.filePath(indexes[0])

        self.ui.OpeningStatusLineEdit.setText("N-03")

        plot_image, self.main.pixmap = readImageAndPixmap(self.main.plot_image_path)
        scene = QGraphicsScene()
        self.main.pixmap_item = scene.addPixmap(self.main.pixmap)

        self.ui.mainImageViewer.setScene(scene)

        self.main.scale = self.ui.scrollAreaImage.width() / plot_image.shape[1]
        self.ui.mainImageViewer.setFixedSize(self.main.scale * self.main.pixmap.size())
        self.ui.mainImageViewer.fitInView(self.main.pixmap_item)
    
    def closeImage(self):
        self.main.plot_image_path = ""

        self.main.scale = 1.0
        self.main.pixmap = None
        self.main.pixmap_item = None

        self.ui.mainImageViewer.setScene(None)
    
    def addImage(self):
        readFilePath = QFileDialog.getOpenFileNames(
                caption="Add images to current working directory", filter="Images (*.png *.jpg)"
                )
        new_image_paths = readFilePath[0]

        if not new_image_paths:
            return

        for new_image_path in new_image_paths:
            src_path = new_image_path

            new_image_name = os.path.basename(new_image_path)
            dst_path = os.path.join(self.main.image_save_folder, new_image_name)

            shutil.copy(src_path, dst_path)
        
            if self.main.plot_image_path != "":
                index_to_select = self.main.fileModel.index(self.main.plot_image_path)

                if index_to_select.isValid():
                    self.ui.ImagetreeView.scrollTo(index_to_select)
                    self.ui.ImagetreeView.setCurrentIndex(index_to_select)
    
    def deleteImage(self):
        if self.main.plot_image_path == "":
            print("선택된 이미지 없음.")
        else:
            os.remove(self.main.plot_image_path)
            self.ui.ImagetreeView.selectionModel().clear()
            self.closeImage()