import cv2
import numpy as np

# 读取图像
img = cv2.imread('_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-17.tif', 0)

# 二值化图像
ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# 寻找图像中的颗粒 (在这里，我们将颗粒视为连通区域)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空图像用于绘制凸壳
hull_image = np.zeros(img.shape, np.uint8)

for i in range(len(contours)):
    # 计算每个颗粒的凸壳
    hull = cv2.convexHull(contours[i])
    
    # 绘制凸壳
    cv2.drawContours(hull_image, [hull], -1, (255, 255, 255), 2)

# 将凸壳图像与原图像叠加
result = cv2.merge((img, img, img))
result = cv2.addWeighted(result, 0.7, cv2.cvtColor(hull_image, cv2.COLOR_GRAY2BGR), 0.3, 0)

cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()