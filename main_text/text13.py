import cv2
import numpy as np
import matplotlib.pyplot as plt

# 合并较小相邻轮廓函数.........................................................

def merge_regions(contours, threshold_distance, min_area):
    # 创建一个空的字典，用于存储合并后的区域
    merged_regions = {}

    # 计算区域之间的距离或重叠度，并进行合并
    for i in range(len(contours)):
        contour_i = contours[i]
        merged = False

        # 检查当前区域的面积是否小于阈值
        area_i = cv2.contourArea(contour_i)
        if area_i < min_area:
            continue

        for j in range(i+1, len(contours)):
            contour_j = contours[j]

            # 计算两个轮廓的形状距离
            distance = cv2.matchShapes(
                contour_i, contour_j, cv2.CONTOURS_MATCH_I1, 0)

            # 检查是否是较小的颗粒合并到最近的大颗粒
            area_j = cv2.contourArea(contour_j)

            if area_i < area_j and distance < threshold_distance:
                merged_regions[j] = merged_regions.get(j, []) + [contour_i]
                merged = True
            elif area_i > area_j and distance < threshold_distance:
                merged_regions[i] = merged_regions.get(i, []) + [contour_j]
                merged = True

        # 如果该区域没有被合并，则将其作为独立的区域存储
        if not merged:
            merged_regions[i] = merged_regions.get(i, []) + [contour_i]

    return merged_regions

# 合并重叠轮廓函数.........................................................

def remove_overlapping_contours(contours):
    # 计算每个轮廓的边界框
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # 标记需要删除的轮廓索引
    contours_to_remove = []

    # 检查每对边界框是否有重叠
    for i in range(len(bounding_boxes)):
        for j in range(i+1, len(bounding_boxes)):
            box_i = bounding_boxes[i]
            box_j = bounding_boxes[j]

            # 计算边界框的重叠区域
            intersection_x = max(box_i[0], box_j[0])
            intersection_y = max(box_i[1], box_j[1])
            intersection_w = min(
                box_i[0]+box_i[2], box_j[0]+box_j[2]) - intersection_x
            intersection_h = min(
                box_i[1]+box_i[3], box_j[1]+box_j[3]) - intersection_y

            # 判断是否有重叠，越大颗粒越多
            if intersection_w > 10 and intersection_h > 11:
            # if intersection_w > 20 and intersection_h > 20:
            # if intersection_w > 15 and intersection_h > 16:
            # if intersection_w > 25 and intersection_h > 25:
                
                # 标记需要删除的轮廓索引
                contours_to_remove.append(i)
                break

    # 根据标记删除轮廓
    cleaned_contours = [cnt for i, cnt in enumerate(
        contours) if i not in contours_to_remove]

    return cleaned_contours

# 分割算法..................................................................


def watershed_algorithm(image):
    src = image.copy()

    # 边缘保留滤波EPF去噪，sp、sr分别表示空间窗口大小、色彩空间窗口大小
    blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)

    # 转成灰度图像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 使用otsu算法得到二值图像区间阈值
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 进行开运算（先腐蚀后膨胀）
    kernel = np.ones((5, 5), np.uint8)
    smoothed_image = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

    # 对距离变换图像进行多个阈值化处理，获得多个分割结果
    # threshold_values = [0.14, 0.24, 0.28, 0.46, 0.35]  # 自定义多个阈值17nologo
    threshold_values = [0.23, 0.29, 0.37, 0.42, 0.43,0.5]  # 自定义多个阈值7nologo
    # threshold_values = [0.28, 0.19, 0.21, 0.41,
    #                     0.51, 0.52, 0.597]  # 自定义多个阈值10nologo
    # threshold_values = [0.14, 0.25, 0.32, 0.36,
    #                     0.4035, 0.469, 0.49, 0.65, 0.6937]  # 自定义多个阈值19nologo
    # threshold_values = threshold_values = [
    #     0.23, 0.3, 0.44, 0.55, 0.65, 0.8]  # 自定义多个阈值20nologo,0.38
    # threshold_values = [0.23, 0.29,0.35, 0.37, 0.42, 0.43,0.5]  # 自定义多个阈值19nologo

    # threshold_values = threshold_values = [
    #     0.886, 0.65, 0.3, 0.23, 0.55, 0.44,]  # 自定义多个阈值13nologo,0.38
    # threshold_values = threshold_values = [
    #     0.512, 0.4614, 0.14, 0.17, 0.27]  # 自定义多个阈值5nologo
    # threshold_values = [0.7009, 0.3,]  # 自定义多个阈值15nologo
    multiplied_values = [value * dist_out.max() for value in threshold_values]
    print(multiplied_values)
    print(dist_out.max())

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
            binary, cv2.MORPH_DILATE, kernel, iterations=10)
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
    # 获取所有轮廓
    contours = []
    for i in range(2, int(max_val + 1)):
        if i != len(threshold_values):
            ret, thres1 = cv2.threshold(
                markers_8u, i - 1, 255, cv2.THRESH_BINARY)
            ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
            mask = thres1 - thres2
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours.extend(cnts)

    print(len(contours))

# 较小区域合并.............................................
    merged_regions = merge_regions(contours, 0, 10)
    print(len(merged_regions))
# 重叠轮廓合并............................................................................
    # # 合并重叠轮廓
    # # 提取所有轮廓
    all_contours = []
    for contours in merged_regions.values():
        all_contours.extend(contours)

    # 调用函数去除重叠的轮廓
    merged_regions = remove_overlapping_contours(all_contours)

    # 将合并后的轮廓重新组织为字典格式
    non_overlapping_regions = {}
    for i, contour in enumerate(merged_regions):
        non_overlapping_regions[i] = [contour]
    print(len(non_overlapping_regions))
# 分割可视化（区域）...............................................................................
    height, width = image.shape[:2]
    black_background = np.zeros((height, width, 3), dtype=np.uint8)

    for i, contours in non_overlapping_regions.items():
        color = colors[i % len(colors)]
        for contour in contours:
            cv2.drawContours(black_background, [
                             contour], -1, color, thickness=cv2.FILLED)
    result = cv2.addWeighted(black_background, 0.5, src, 0.5, 0)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 分割可视化（红线）...........................................................................
    for i, contours in non_overlapping_regions.items():
        color = colors[i % len(colors)]
        for contour in contours:
                        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

    result = cv2.addWeighted(image, 0.5, src, 0.5, 0)
    # cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 计算等效直径，真实尺寸0.02324219微米/像素.................................................
    equivalent_diameters = []
    for i, contours in non_overlapping_regions.items():
        for contour in contours:
            area = cv2.contourArea(contour)
            equivalent_diameter = 2*np.sqrt(area / np.pi)*0.02324219
            equivalent_diameters.append(equivalent_diameter)
    print(len(equivalent_diameters))
# 将等效直径在图像中显示....................................................................
    for i, contours in non_overlapping_regions.items():
        color = colors[i % len(colors)]
        for contour in contours:
            # 计算等效直径
            area = cv2.contourArea(contour)
            equivalent_diameter = 2 * np.sqrt(area / np.pi)*0.02324219

            # 绘制等效直径文本
            if 0 <= equivalent_diameter <= 0.48:
                (x, y), _ = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                cv2.putText(black_background, f'{equivalent_diameter:.2f}', center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
# 绘制结果图..................................................................................
    result = cv2.addWeighted(black_background, 0.5, src, 0.5, 0)
    # cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 绘制粒径分布直方图......................................................................
    # 绘制粒径分布直方图
    bins = [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.8]
    plt.hist(equivalent_diameters, bins=bins, edgecolor='black')
    plt.xlabel('Equivalent Diameter')
    plt.ylabel('Count')
    plt.title('Particle Size Distribution')
    plt.xticks(bins)
    # plt.show()


src = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-07.png')
# src = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-10.png')
# src = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-19.png')
# src = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-20.png')
# src = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-12.png')

segmented_image = watershed_algorithm(src)
