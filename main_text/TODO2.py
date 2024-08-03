import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 读取图像
# 读取二值图像
image = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-19.png')
blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
# 计算灰度值分布
hist, bins = np.histogram(dist_out.flatten(), bins=256, range=[0.001, 0.03])

# 绘制灰度曲线图
plt.plot(bins[:-1], hist)

# 寻找极大值点
maxima_indices = argrelextrema(hist, np.greater)[0]

# 设置阈值来选择频率较高的极大值点
threshold = np.percentile(hist, 90)
selected_maxima_indices = [
    index for index in maxima_indices if hist[index] > threshold]

# 在图像上标记选择的极大值点并打印x坐标
for index in selected_maxima_indices:
    x = bins[index]
    y = hist[index]
    plt.scatter(x, y, c='r', marker='o')
    print(x/dist_out.max())

# 设置坐标轴标签和标题
plt.xlabel('Gray Level')
plt.ylabel('Pixel Count')
plt.title('Gray Level Distribution of Distance Transform')

# 显示图像
plt.show()

# # 计算直方图
# histogram = cv2.calcHist([dist_out], [0], None, [125], [0, 256])

# # 可视化直方图
# plt.hist(dist_out.ravel(), bins=len(histogram), range=[0.001, 0.03], color='b', alpha=0.5)
# print(len(histogram))

# plt.title('Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
