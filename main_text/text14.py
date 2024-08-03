import cv2
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

def watershed_algorithm(image):
    src = image.copy()

    # 边缘保留滤波EPF去噪，sp、sr分别表示空间窗口大小、色彩空间窗口大小
    blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)

    # 转成灰度图像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 使用otsu算法得到二值图像区间阈值
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

    threshold_values = [0.23, 0.29,  0.37,
                        0.5, 0.42, 0.43]  # 自定义多个阈值7nologo
    # threshold_values = [0.28, 0.19, 0.21, 0.41,
    #                     0.51, 0.52, 0.597]  # 自定义多个阈值10nologo
    # threshold_values = [0.14, 0.25, 0.32, 0.36,
    #                     0.4035, 0.469, 0.49, 0.65, 0.6937]  # 自定义多个阈值19nologo
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
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_DILATE, kernel, iterations=1)
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
    # 初始化一个字典来存储合并后的区域
    merged_regions = {}
    for i in range(2, int(max_val + 1)):
        ret, thres1 = cv2.threshold(markers_8u, i - 1, 255, cv2.THRESH_BINARY)
        ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
        mask = thres1 - thres2
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 忽略背景
        if i != len(threshold_values):
            cv2.drawContours(image, contours, -1, colors[(i - 2) % len(colors)], -1)
            cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    # 将原图和分割后的图像进行叠加，得到可视化显示的结果
    result = cv2.addWeighted(src, 0.5, image, 0.5, 0)  # 图像权重叠加
    cv2.imshow('result', result)

    # 显示结果图像
    # cv2.imshow('edge_image', image)

    # 显示结果图像
    # cv2.imshow('Segmented Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


src = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-07.png')
# src = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-10.png')
# src = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-19.png')
segmented_image = watershed_algorithm(src)
