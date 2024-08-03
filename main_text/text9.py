import cv2
import numpy as np

from scipy import ndimage as ndi

def merge_small_areas(combined_markers, min_size):
    # 标记连通区域
    labeled, num_objects = ndi.label(combined_markers)
    sizes = np.bincount(labeled.ravel())
    
    # 创建一个新的标记图像
    new_markers = np.zeros_like(labeled)
    
    # 对每个连通区域进行处理
    for i in range(1, num_objects + 1):
        # 如果这个区域太小，那我们就找到最近的邻居，并合并
        if sizes[i] < min_size:
            # 创建一个二值图像，只包含这个连通区域
            blob = np.where(labeled == i, 1, 0)
            
            # 使用膨胀操作找到最近的邻居
            dilated = cv2.dilate(blob, np.ones((3,3)), iterations=1)
            neighbors = np.where(dilated - blob > 0, labeled, 0)
            
            # 找到最常见的邻居（即最近的大颗粒）
            counts = np.bincount(neighbors[neighbors > 0].ravel())
            most_common_neighbor = np.argmax(counts)
            
            # 将小颗粒合并到最常见的邻居
            new_markers[labeled == i] = most_common_neighbor
        else:
            new_markers[labeled == i] = i
            
    return new_markers

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
    threshold_values = [0.14,0.18,0.24]  # 自定义多个阈值17
    # threshold_values = [0.1,0.14,0.26] # 自定义多个阈值20
    # threshold_values = [0.14,0.21] # 自定义多个阈值19
    # threshold_values = [0.18,0.28,0.08] # 自定义多个阈值07
    # threshold_values = [0.1,0.11,0.17,0.2,0.22,0.24,] # 自定义多个阈值10
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
    
    # 调用merge_small_areas函数合并小区域
    min_size = 100  # 最小尺寸阈值
    combined_markers = merge_small_areas(combined_markers, min_size)

    # 将分割结果可视化显示在原图上
    min_val, max_val, _, _ = cv2.minMaxLoc(combined_markers)
    # print(max_val)
    markers_8u = np.uint8(combined_markers)
    cv2.imshow('markers_8u', markers_8u)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
              (255,0,255), (0,255,255), (255,128,0), (255,0,128),
              (128,255,0), (128,0,255), (255,128,128), (128,255,255)]
    for i in range(2, int(max_val + 1)):
        ret, thres1 = cv2.threshold(markers_8u, i - 1, 255, cv2.THRESH_BINARY)
        ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
        mask = thres1 - thres2
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        

        # 忽略背景
        if i != len(threshold_values):
            cv2.drawContours(image, contours, -1, colors[(i - 2) % len(colors)], -1)
        
        # # 计数
        # for contour in contours:
        #     M = cv2.moments(contour)

        #     # Check if the moment 'm00' is non-zero
        #     if M["m00"] != 0:
        #         centroid_x = int(M["m10"] / M["m00"])
        #         centroid_y = int(M["m01"] / M["m00"])
        #         cv2.putText(image, str(i - 1), (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        #                     cv2.LINE_AA)
    
    # 将原图和分割后的图像进行叠加，得到可视化显示的结果
    result = cv2.addWeighted(src, 0.5, image, 0.5, 0)  # 图像权重叠加
    cv2.imshow('result', result)

    # 在原图中添加区域数量的文字标注
    # cv2.putText(image, "count=%d" % (int(max_val - 1)), (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示结果图像
    cv2.imshow('Segmented Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


src = cv2.imread('_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-17.tif')
watershed_algorithm(src)