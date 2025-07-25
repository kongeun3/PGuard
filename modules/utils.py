
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap
from skimage import io

def cvtArrayToQImage(array: np.array) -> QImage:

    if len(array.shape) == 3 : 

        h, w, c = array.shape
        if c == 3:
            return QImage(array.data, w, h, 3 * w, QImage.Format_RGB888)
        elif c == 4: 
            return QImage(array.data, w, h, 4 * w, QImage.Format_RGBA8888)

    elif len(array.shape) == 2 :
        h, w = array.shape
        return QImage(array.data, w, h, QImage.Format_Mono)
    
def readImageAndPixmap(path):
        image = np.array(Image.open(path))
        return image, QPixmap(cvtArrayToQImage(image))

def compute_iou_np(box, boxes):
    """box: (4,), boxes: (N, 4)"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou

def nms_numpy(bboxes, scores, iou_threshold=0.5):
    """bboxes: (N, 4), scores: (N,)"""
    indices = np.argsort(scores)[::-1]  # score 내림차순 정렬
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        current_box = bboxes[current]
        other_boxes = bboxes[indices[1:]]
        ious = compute_iou_np(current_box, other_boxes)
        indices = indices[1:][ious <= iou_threshold]

    return keep