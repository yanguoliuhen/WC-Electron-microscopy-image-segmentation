import cv2
import numpy as np
import matplotlib.pyplot as plt

def h_maxima_transform(image, h_value):
    # 计算距离变换
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)

    # 应用H-maxima变换
    h_maxima_transformed = np.maximum(dist_transform - h_value, 0)

    return h_maxima_transformed

def watershed_segmentation(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化图像（可以根据具体情况选择合适的阈值方法）
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 使用H-maxima变换
    h_value = 0.0005
    h_maxima_result = h_maxima_transform(binary_image, h_value)

    # 转换为8位无符号整数
    h_maxima_result = np.uint8(h_maxima_result)

    # 使用分水岭算法进行分割
    _, markers = cv2.connectedComponents(h_maxima_result)
    markers = markers + 1
    markers[~binary_image] = 0  # 将背景标记为0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # 将分水岭线标记为红色

    return image

# 读取图像
image = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-07.png')

# 进行分割
segmented_image = watershed_segmentation(image)

# 显示结果
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(122)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image')

plt.show()
