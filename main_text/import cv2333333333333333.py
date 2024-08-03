import cv2
import numpy as np
import matplotlib.pyplot as plt


def are_regions_similar(region1, region2, threshold_distance=10000):
    # 比较两个区域的位置
    # 我们比较了质心之间的距离
    M1 = cv2.moments(region1)
    cx1 = int(M1["m10"] / M1["m00"])
    cy1 = int(M1["m01"] / M1["m00"])

    M2 = cv2.moments(region2)
    cx2 = int(M2["m10"] / M2["m00"])
    cy2 = int(M2["m01"] / M2["m00"])

    distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    return distance <= threshold_distance


# 读取图像
image = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-10.png')

src = image.copy()

# 边缘保留滤波EPF去噪，sp、sr分别表示空间窗口大小、色彩空间窗口大小
blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)

# 转成灰度图像
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# 使用otsu算法得到二值图像区间阈值
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 距离变换
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

# threshold_values = [0.06,0.1,0.17,0.2,0.26] # 自定义多个阈值10withlogo
threshold_values = [0.2, 0.21, 0.41, 0.51, 0.52, 0.597]  # 自定义多个阈值10nologo
multiplied_values = [value * dist_out.max() for value in threshold_values]
print(multiplied_values)

segmented_images = []
for threshold in threshold_values:
    _, surface = cv2.threshold(
        dist_out, threshold * dist_out.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(surface)

    # 使用连通组件分析算法，得到初始标记图像
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1

    # 对二值图像进行膨胀处理，得到未知区域的标记
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    unknown = binary - sure_fg

    # 未知区域标记为0
    markers[unknown == 255] = 0

    # 分水岭算法分割
    markers = cv2.watershed(image, markers=markers)
    segmented_images.append(markers)

# 将所有分割结果叠加在一起
combined_markers = np.zeros_like(segmented_images[0])
for markers in segmented_images:
    combined_markers += markers

# 将分割结果可视化显示在原图上
min_val, max_val, _, _ = cv2.minMaxLoc(combined_markers)
markers_8u = np.uint8(combined_markers)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 128),
          (128, 255, 0), (128, 0, 255), (255, 128, 128), (128, 255, 255)]
# edge_image = src.copy()
# 初始化一个字典来存储合并后的区域
merged_regions = {}
for i in range(2, int(max_val + 1)):
    ret, thres1 = cv2.threshold(markers_8u, i - 1, 255, cv2.THRESH_BINARY)
    ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
    mask = thres1 - thres2
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
# 忽略背景
    for contour in contours:
        # for j in contour:
        # 计算每个等值线的面积
        area = cv2.contourArea(contour)

        # 为要保留的区域的最小区域设置阈值
        min_area_threshold = 1

        if area >= min_area_threshold:
            # 绘制面积高于阈值的区域的等值线
            cv2.drawContours(image, [contour], -1,
                             colors[(i - 2) % len(colors)], -1)

            # 将合并的区域存储在字典中
            if i in merged_regions:
                merged_regions[i].append(contour)
            else:
                merged_regions[i] = [contour]

print(len(merged_regions))

# 创建一个新字典来存储唯一区域
unique_regions = {}

# 遍历merged_regions字典
for threshold, regions in merged_regions.items():
    # 创建一个列表来存储当前阈值的唯一区域
    unique_regions[threshold] = []

    # 遍历当前阈值的区域
    for region in regions:
        # 检查当前区域是否与任何唯一区域相似
        is_similar = False
        for unique_region in unique_regions[threshold]:
            if are_regions_similar(region, unique_region):
                is_similar = True
                break

        # 如果区域与任何唯一区域不相似，请将其添加到列表中
        if not is_similar:
            unique_regions[threshold].append(region)
print(len(unique_regions))

# for key in unique_regions:
#     for contour in unique_regions[key]:

#         # print(len(contour))

# # for key in unique_regions:
# #     for contour in unique_regions[key]:
# #         cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)  # 红色线条，线宽1
# # result = cv2.addWeighted(src, 0.5, image, 0.5, 0)  # 图像权重叠加
# # cv2.imshow('edge_image', image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ecd_values = []
# for key in unique_regions:
#     for contour in unique_regions[key]:
#         # 绘制面积标签在粒子上
#         M = cv2.moments(contour)
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         cv2.drawMarker(image, (cx, cy), (0, 0, 255), 1, 10, 2)
#         area = cv2.contourArea(contour)
#         ecd = np.sqrt((4 * area) / np.pi)
#         ecd_values.append(ecd)

# print(len(ecd_values))

# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Plot PSD graph
# plt.figure()
# plt.hist(ecd_values, bins=20,edgecolor='black')
# plt.xlabel('Equivalent Circular Diameter (ECD)')
# plt.ylabel('Count')
# plt.title('Particle Size Distribution (auto)')
# plt.show()
