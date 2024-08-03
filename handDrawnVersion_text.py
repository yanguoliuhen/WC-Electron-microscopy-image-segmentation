import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(
    'Harris_watershed/text_image/2_10.png')

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 对图像进行二值化处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 查找图像中的轮廓
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化粒子数量和面积列表
particle_count = 0
particle_areas = []
particle_diameters = []

# 遍历每个轮廓
for contour in contours:
    # 计算轮廓的面积
    area = cv2.contourArea(contour)

    # 根据面积大小判断是否为粒子
    if 10000 > area > 1:
        # 绘制面积标签在粒子上
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # cv2.drawMarker(image, (cx, cy), (0, 0, 255), 1, 10, 2)

        # 计算等效圆直径
        equivalent_diameter = 2 * np.sqrt(area / np.pi)*0.02324219
        particle_diameters.append(equivalent_diameter)
        # # 绘制等效直径文本
        # cv2.putText(image, f'{equivalent_diameter:.2f}', (cx, cy),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # particle_diameters.append(equivalent_diameter)
        # 在图片上绘制等效直径 
        if 0 <= equivalent_diameter <= 200:
            # 绘制面积标签在粒子上
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawMarker(image, (cx, cy), (0, 0, 255), 1, 5, 1)

            # 绘制等效直径文本
            cv2.putText(image, f'{equivalent_diameter:.2f}', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 增加粒子数量和直径到列表中
        particle_count += 1
        particle_areas.append(equivalent_diameter)

# 输出粒子数量和平均面积
print("Particle Count:", particle_count)
print("Average Area:", sum(particle_areas) / len(particle_areas))

# 显示带有面积标签的图像
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 绘制粒度分布图
# plt.hist(particle_diameters, bins=10, edgecolor='black')
# plt.xlabel('Equivalent Diameter')
# plt.ylabel('Count')
# plt.title('Particle Size Distribution (hand)')
# plt.show()
# Plot PSD graph
# 绘制粒径分布直方图
bins = [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.8]
# 绘制粒径分布直方图
plt.hist(particle_diameters, bins=bins, edgecolor='black')
plt.xlabel('Equivalent Diameter')
plt.ylabel('Count')
plt.title('Particle Size Distribution (hand)')
plt.xticks(bins)
plt.show()
