import os

from PIL import Image
import numpy as np
import pandas as pd
import cv2

from mmdet.apis import init_detector, inference_detector
from mmpretrain import ImageClassificationInferencer

from PySide6.QtWidgets import QFileDialog, QStyledItemDelegate, QHeaderView
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtGui import QColor

from .utils import nms_numpy

class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            return str(value)
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        else:
            return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self._df.columns[section]
        elif orientation == Qt.Vertical:
            return str(self._df.index[section])


class HoverRowDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hovered_row = -1

    def setHoveredRow(self, row):
        self._hovered_row = row

    def paint(self, painter, option, index):
        if index.row() == self._hovered_row:
            painter.save()
            painter.fillRect(option.rect, QColor(189, 147, 249))
            painter.restore()
        super().paint(painter, option, index)


class AIClass():
    def __init__(self, ui, main_window):
        # Inherit Class
        self.ui = ui
        self.main = main_window

        # Load AI Model
        GDINO_CONFIG_PATH = os.path.join(self.main.home, "init", "dnn", "config", "grounding_dino.py")
        GDINO_CHECKPOINT_PATH = os.path.join(self.main.home, "init", "dnn", "checkpoint", "grounding_dino.pth")
        self.main.gdino_model = init_detector(GDINO_CONFIG_PATH, GDINO_CHECKPOINT_PATH, device=self.main.device)

        RESNET_CONFIG_PATH = os.path.join(self.main.home, "init", "dnn", "config", "resnet.py")
        RESNET_CHECKPOINT_PATH = os.path.join(self.main.home, "init", "dnn", "checkpoint", "resnet.pth")
        self.main.resnet_model = ImageClassificationInferencer(
            model=RESNET_CONFIG_PATH,
            pretrained=RESNET_CHECKPOINT_PATH,
            device=self.main.device
        )

        # Function Connection
        self.ui.ImageFolderButton.clicked.connect(self.selectImageFolder)
        self.ui.ImageFolderButton.clicked.connect(self.saveResult)
        self.ui.gdinoThresholdButton.clicked.connect(self.changethr)

        # Initialize Class
        print("load ai class")
    
    def selectImageFolder(self):
        # Select Folder
        self.main.image_src_folder = str(QFileDialog.getExistingDirectory(self, "Select Image Folder"))
        self.ui.ImageFolderLineEdit.setText(str(self.main.image_src_folder))

        # Initialize Result Dict
        self.ai_result_dict = {"image": [], "image_path": [], "dino_bbox": [], "dino_score": [], "resnet": []}

        # Call AI Calculation
        self.readImageFolder()
    
    def readImageFolder(self):
        image_list = [file for file in os.listdir(self.main.image_src_folder) if file.endswith((".jpg", ".png"))]

        for image_name in image_list:
            # Load Image
            image_path = os.path.join(self.main.image_src_folder, image_name)
            image = np.array(Image.open(image_path))
            self.main.ai_result_dict["image"].append(image_name)
            self.main.ai_result_dict["image_path"].append(image_path)

            # Grounding DINO
            dino_result = inference_detector(self.main.gdino_model, image, text_prompt=self.main.text_prompt)

            bboxes = dino_result.pred_instances.bboxes.cpu().numpy()
            scores = dino_result.pred_instances.scores.cpu().numpy()
            labels = dino_result.pred_instances.labels.cpu().numpy()

            filtered_bboxes = bboxes[scores > self.main.score_thr]
            filtered_scores = scores[scores > self.main.score_thr]
            filtered_labels = labels[scores > self.main.score_thr]

            keep_indices = nms_numpy(filtered_bboxes, filtered_scores, iou_threshold=0.5)
            nms_bboxes = filtered_bboxes[keep_indices]
            nms_scores = filtered_scores[keep_indices]
            nms_labels = filtered_labels[keep_indices]

            self.main.ai_result_dict["dino_bbox"].append(nms_bboxes)
            self.main.ai_result_dict["dino_score"].append(nms_scores)

            # ResNet
            if len(nms_bboxes) >= 1:
                resnet_result = self.main.resnet_model(image)
                resnet_class = resnet_result[0]['pred_class']
                if resnet_class == 'Y-03':
                    final_cls = 1
                elif resnet_class == 'N-03':
                    final_cls = 2
                else:
                    final_cls = 0
            else:
                final_cls = 0
            
            self.main.ai_result_dict["resnet"].append(final_cls)
        
        # Call Plot Function
        print("Calculation Complete")
        self.showTable()
    
    def showTable(self):
        # Definition Part
        classification_map = {0: "개구부 없음", 1: "정상", 2: "비정상(열림)"}

        # Make DataFrame
        score_list_for_df = []
        for scores in self.main.ai_result_dict["dino_score"]:
            if len(scores) >= 1:
                score = str(round(np.max(scores) * 100, 2))
            else:
                score = "0.00"
            score_list_for_df.append(score)
            
        dict_for_df = {
            "번호(No.)": list(range(1, len(self.main.ai_result_dict["image"]) + 1)),
            "이미지 이름": self.main.ai_result_dict["image"],
            "분류": [classification_map[cls_code] for cls_code in self.main.ai_result_dict["resnet"]],
            "개구부 존재 확률(%)": score_list_for_df,
            "이미지 경로": self.main.ai_result_dict["image_path"]
        }

        # Show DataFrame
        df_for_show = pd.DataFrame(dict_for_df)

        model = DataFrameModel(df_for_show)
        self.ui.ImageTableView.setModel(model)
        
        self.ui.ImageTableView.verticalHeader().setVisible(False)
        self.ui.ImageTableView.horizontalHeader().setMinimumSectionSize(100)
        self.ui.ImageTableView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.ui.ImageTableView.resizeColumnsToContents()
    
    def clearTable(self):
        self.ui.ImageTableView.setModel(None)
        self.main.ai_result_dict = {"image": [], "image_path": [], "dino_bbox": [], "dino_score": [], "resnet": []}
    
    def saveResult(self):
        if len(self.main.ai_result_dict["image"]) <= 0:
            print("모델 연산 결과가 없습니다.")
        else:
            # 1. Make New Dict(resnet=2)
            target_idxs= [i for i, value in enumerate(self.main.ai_result_dict["resnet"]) if value == 2]
            filtered_dict = {
                key: [value_list[i] for i in target_idxs]
                for key, value_list in self.main.ai_result_dict.items()
            }
            print(filtered_dict)

            # 2. Save Images
            for idx, image_name in enumerate(filtered_dict["image"]):
                # Read Data
                src_image_path = filtered_dict["image_path"][idx]
                image = np.array(Image.open(src_image_path))
                bboxes = filtered_dict["dino_bbox"][idx]
                scores = filtered_dict["dino_score"][idx]

                # Plot
                plot_image = image.copy()

                for bbox, score in zip(bboxes, scores):
                    x1, y1, x2, y2 = bbox.astype(int)
                    
                    text = f'비정상 개구부 {score:.2f}'
                    
                    cv2.rectangle(plot_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(plot_image, (x1, y1 - 25), (x1 + w, y1 - 5), (0, 0, 255), -1)
                    cv2.putText(plot_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Write Image
                dst_image_path = os.path.join(self.main.image_save_folder, image_name)
                Image.fromarray(plot_image).save(dst_image_path)
            
            # 3. Initialization
            self.InitializeAIFunc()
        
    def InitializeAIFunc(self):
        self.image_src_folder = ""
        self.ui.ImageFolderLineEdit.setText(str(self.main.image_src_folder))
        self.clearTable()
    
    def changethr(self):
        self.main.score_thr = float(self.ui.gdinoThresholdLineEdit.text())
        self.ui.gdinoThresholdLabel.setText(f"Current Threshold: {self.main.score_thr}")
        self.ui.gdinoThresholdLineEdit.setText("")
