import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_equivalent_diameter(contour, pixel_size):
    # 计算等效直径（以微米为单位）
    area = cv2.contourArea(contour)
    equivalent_diameter = np.sqrt(4 * area / np.pi) * pixel_size
    return equivalent_diameter

# 读取已分割完毕的图像
image_path = "Harris_watershed/text_image/2_10.png"  # 替换成你的图像路径
img = cv2.imread(image_path)

# 将图像转换为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 查找轮廓
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 真实尺寸每像素0.02324219微米
pixel_size = 0.02324219

# 创建一个空白图像用于绘制轮廓
contour_img = np.zeros_like(img)

# 计算每个颗粒的等效直径并绘制轮廓
for contour in contours:
    equivalent_diameter = calculate_equivalent_diameter(contour, pixel_size)
    cv2.drawContours(contour_img, [contour], 0, (255, 255, 255), 1)  # 绘制白色轮廓

# 显示原始图像和带有轮廓的图像
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(contour_img, cmap='gray'), plt.title('Contours')
plt.show()

# 绘制PSD分布直方图
equivalent_diameters = [calculate_equivalent_diameter(contour, pixel_size) for contour in contours]
# plt.hist(equivalent_diameters, bins=10, edgecolor='black')
# plt.xlabel('Equivalent Diameter (micrometers)')
# plt.ylabel('Frequency')
# plt.title('Particle Size Distribution (PSD)')
# plt.show()
# 绘制粒径分布直方图
bins = [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.8]
# 绘制粒径分布直方图
plt.hist(equivalent_diameters, bins=bins, edgecolor='black')
plt.xlabel('Equivalent Diameter')
plt.ylabel('Count')
plt.title('Particle Size Distribution (hand)')
plt.xticks(bins)
plt.show()


