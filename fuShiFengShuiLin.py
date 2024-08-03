import cv2
import numpy as np

# 读取图像
image = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-07.png')

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 阈值分割
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 膨胀操作，以填充目标区域的小孔洞
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(threshold, kernel, iterations=2)

# 腐蚀操作
eroded = cv2.erode(dilated, kernel, iterations=1)

# 标记分水岭区域
markers = np.zeros_like(gray, dtype=np.int32)
markers[eroded == 255] = 1

# 应用分水岭算法
cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # 在分水岭线上绘制红色边界

# 显示并保存结果
cv2.imshow('Segmented Image', image)
cv2.imwrite('segmented_image.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()