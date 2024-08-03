import cv2
import matplotlib.pyplot as plt

# 读取距离变换图像
image = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-10.png')
blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

# 提取像素值
pixel_values = dist.flatten()

# 统计像素值频率
pixel_counts = {}
for pixel in pixel_values:
    if pixel in pixel_counts:
        pixel_counts[pixel] += 1
    else:
        pixel_counts[pixel] = 1

# 排序像素值和频率
sorted_pixels = sorted(pixel_counts.items())
pixels, counts = zip(*sorted_pixels)

# 绘制直方图
plt.bar(pixels, counts)

# 设置图表标题和轴标签
plt.title('Pixel Histogram of Distance Transform Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# 显示直方图
plt.show()