import cv2
import numpy as np

def watershed_algorithm(image):
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

    # 对距离变换图像进行多个阈值化处理，获得多个分割结果
    # threshold_values = [0.14, 0.24]  # 自定义多个阈值17
    # threshold_values = [0.1,0.14,0.26] # 自定义多个阈值20
    threshold_values = [0.14,0.21] # 自定义多个阈值19
    # threshold_values = [0.14] # 自定义多个阈值18
    segmented_images = []
    for threshold in threshold_values:
        _, surface = cv2.threshold(dist_out, threshold * dist_out.max(), 255, cv2.THRESH_BINARY)
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
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
              (255,0,255), (0,255,255), (255,128,0), (255,0,128),
              (128,255,0), (128,0,255), (255,128,128), (128,255,255),
              (0,128,255), (0,255,128), (128,0,128), (128,128,255),
              (255,255,255), (0,0,0)]

    image_copy = image.copy()
    contours = []
    for i in range(2, int(max_val)):
        if i - 1 != 1:  # Skip background region
            mask = np.where(markers_8u == i, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_copy, contours, -1, colors[(i - 2) % len(colors)], -1)

        # 计数
        for contour in contours:
            M = cv2.moments(contour)

            # Check if the moment 'm00' is non-zero
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                cv2.putText(image_copy, str(i - 1), (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

    # 将原图和分割后的图像进行叠加，得到可视化显示的结果
    result = cv2.addWeighted(src, 0.1, image_copy, 0.9, 0)  # 图像权重叠加

    # 显示结果
    cv2.imshow("result", result)

# 读取图像
image_path = "_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-19.tif"
image = cv2.imread(image_path)

# 应用分水岭算法进行图像分割
watershed_algorithm(image)

# 等待按键退出
cv2.waitKey(0)
cv2.destroyAllWindows()