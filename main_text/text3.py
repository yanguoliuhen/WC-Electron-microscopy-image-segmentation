import cv2
import numpy as np

def non_maximum_suppression(boxes, scores, overlap_threshold=0.3):
    """
    boxes: 每个区域的坐标，格式为[x1, y1, x2, y2]
    scores: 每个区域的置信度，可以是分类器输出的概率值或者其他打分方式
    overlap_threshold: 重叠面积的阈值
    """
    # 计算每个区域的面积
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # 按置信度从大到小排序
    order = np.argsort(scores)[::-1]

    keep = []  # 用于存放保留的区域的索引
    while order.size > 0:
        # 选取置信度最高的区域
        i = order[0]
        keep.append(i)

        # 计算当前区域与其他所有未处理区域的重叠部分的面积
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]

        # 找到重叠面积小于阈值的区域
        inds = np.where(overlap <= overlap_threshold)[0]

        # 更新未处理区域的索引
        order = order[inds + 1]

    return keep

# 读取待分割图像
img = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-07.png')

# 进行图像分割
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 进行形态学操作，填充孔洞
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 找到图像中所有的轮廓
contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 对每个区域进行单独的处理
boxes = []
scores = []
for i, contour in enumerate(contours):
    # 提取当前区域的边界框
    x, y, w, h = cv2.boundingRect(contour)
    # 计算当前区域的置信度（可以是面积、分类器输出的概率值等）
    score = w * h
    boxes.append([x, y, x + w - 1, y + h - 1])
    scores.append(score)

# 进行 NMS 操作，去除重叠的区域
keep = non_maximum_suppression(np.array(boxes), np.array(scores))
for i in keep:
    # 在原图上绘制边界框
    x, y, x2, y2 = boxes[i]
    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
    # 对当前区域进行处理
    roi = gray[y:y + h, x:x + w]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    # 将处理结果覆盖到原图的对应区域上
    img[y:y + h, x:x + w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

# 显示处理结果
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()