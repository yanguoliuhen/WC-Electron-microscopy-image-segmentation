import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-07.png')
src = image.copy()
blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
# threshold_values = [0.2, 0.21, 0.41, 0.51, 0.52, 0.597]  # 自定义多个阈值10nologo
threshold_values = [0.23, 0.29,  0.37, 0.5, 0.42, 0.43]  # 自定义多个阈值07nologo
multiplied_values = [value * dist_out.max() for value in threshold_values]
print(multiplied_values)
segmented_images = []
for threshold in threshold_values:
    _, surface = cv2.threshold(
        dist_out, threshold * dist_out.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(surface)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    unknown = binary - sure_fg
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers=markers)
    segmented_images.append(markers)

# 将所有分割结果叠加在一起
combined_markers = np.zeros_like(segmented_images[0])
for markers in segmented_images:
    combined_markers += markers
visualization = np.zeros_like(image)
# 将分割结果可视化显示在原图上
min_val, max_val, _, _ = cv2.minMaxLoc(combined_markers)
markers_8u = np.uint8(combined_markers)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 128),
          (128, 255, 0), (128, 0, 255), (255, 128, 128), (128, 255, 255)]
for i in range(2, int(max_val + 1)):
    ret, thres1 = cv2.threshold(markers_8u, i - 1, 255, cv2.THRESH_BINARY)
    ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
    mask = thres1 - thres2
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if i != len(threshold_values):
        cv2.drawContours(visualization, contours, -1, (255, 255, 255), -1)
        for contour in contours:
            cv2.drawContours(visualization, [contour], -1, (0, 0, 0), 1)
visualization_gray = cv2.cvtColor(visualization, cv2.COLOR_BGR2GRAY)
cv2.imshow("visualization", visualization)

ret, visualization_thresh = cv2.threshold(visualization_gray, 1, 255, 0)
contours, hierarchy = cv2.findContours(
    visualization_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, colors[(i-2) % 12], -1)
ecd_values = []

for contour in contours:
    # 绘制面积标签在粒子上
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.drawMarker(image, (cx, cy), (0, 0, 255), 1, 10, 2)
    area = cv2.contourArea(contour)
    ecd = np.sqrt((4 * area) / np.pi)
    ecd_values.append(ecd)

print(len(ecd_values))

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Plot PSD graph
plt.figure()
plt.hist(ecd_values, bins=10, edgecolor='black')
plt.xlabel('Equivalent Circular Diameter (ECD)')
plt.ylabel('Count')
plt.title('Particle Size Distribution (auto)')
plt.show()

#     # 显示结果
# cv2.imshow("Segmentation Result", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
