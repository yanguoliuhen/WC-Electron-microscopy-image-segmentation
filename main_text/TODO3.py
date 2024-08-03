import cv2
import numpy as np
from scipy.signal import find_peaks

def find_best_threshold_ratio(image):
    # 计算图像的直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # 计算累积直方图
    cumulative_hist = np.cumsum(hist)

    # 计算累积直方图的梯度
    gradient = np.gradient(cumulative_hist)

    # 寻找波峰位置
    peak_positions, _ = find_peaks(gradient)

    # 寻找波谷位置（负峰值）
    valley_positions, _ = find_peaks(-gradient)

    # 找到波峰和波谷位置之间的阈值比例
    threshold_ratios = []
    for peak in peak_positions:
        # 找到最接近波峰位置的波谷位置
        closest_valley = min(valley_positions, key=lambda x: abs(x - peak))
        # 计算阈值比例
        threshold_ratio = closest_valley / 0.019
        threshold_ratios.append(threshold_ratio)

    return threshold_ratios

# 读取灰度图像
image = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-17.png')
blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

# 寻找最佳的阈值比例
best_threshold_ratios = find_best_threshold_ratio(dist_out)

print("Best threshold ratios:", best_threshold_ratios)
