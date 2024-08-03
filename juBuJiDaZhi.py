import cv2
import numpy as np

# 加载距离图像
image = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-07.png')
# 边缘保留滤波EPF去噪，sp、sr分别表示空间窗口大小、色彩空间窗口大小
blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)
# 转成灰度图像
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# 使用otsu算法得到二值图像区间阈值
ret, binary = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 距离变换
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

# 寻找局部极大值
local_max = cv2.dilate(dist_out, np.ones((3,3),np.uint8))
local_max_mask = np.uint8(dist_out == local_max)

# 将局部极大值标记在新的图像上
markers = local_max_mask * 255

# 反转颜色（黑变白，白变黑）
inverted_markers = 255 - markers

cv2.imshow('inverted_markers', inverted_markers)

cv2.waitKey(0)
cv2.destroyAllWindows()


