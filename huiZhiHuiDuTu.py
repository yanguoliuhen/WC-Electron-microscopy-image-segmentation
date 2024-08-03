import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载图像
image = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-07.png')

# 2. 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 计算灰度直方图
histogram, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

# 4. 计算灰度百分比占比
total_pixels = gray_image.shape[0] * gray_image.shape[1]
percentage = (histogram / total_pixels) * 100

# 5. 绘制直方图
plt.figure()
plt.plot(percentage, color='black')
plt.xlabel('灰度级别')
plt.ylabel('百分比')
plt.title('灰度直方图')
plt.grid(True)

# 6. 显示直方图
plt.show()
