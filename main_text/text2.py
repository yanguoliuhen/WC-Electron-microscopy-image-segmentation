import cv2
import numpy as np

# 加载图像
image = cv2.imread("_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-17.tif")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值处理将图像转换为二值图像
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# 执行凸壳分割
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制凸壳分割结果
result = np.zeros_like(image)
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# 显示分割结果
cv2.imshow("Segmented Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()