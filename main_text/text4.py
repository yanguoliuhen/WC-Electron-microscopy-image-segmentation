import cv2
import numpy as np
import random as rd

# 定义自定义函数
def watershed_algorithm(image):
   

    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow('distance-Transform', dist_out * 100)

    # 对距离变换图像进行二值化处理，得到前景区域的初始标记，0.5*dist_out.max()表示阈值
    # ret, surface = cv2.threshold(dist, 0.14*dist.max(), 255, cv2.THRESH_BINARY)
    # 使用局部均值法确定距离变换图像的最佳标记阈值
    window_size = 9
    local_mean = cv2.blur(dist_out, (window_size, window_size))
    mask = dist_out > local_mean
    threshold_value = dist_out[mask].mean()
    ret, surface = cv2.threshold(dist_out, threshold_value, 255, cv2.THRESH_BINARY)

    # 转成8位整型
    sure_fg = np.uint8(surface)

    # 使用连通组件分析算法，得到初始标记图像，其中前景区域的标记值为2，背景区域的标记值为1
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 #整个图+1，使背景不是0而是1值
    
    # 对二值图像进行膨胀处理，得到未知区域的标记。将未知区域的标记值设为0
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    unknown = binary - sure_fg

    # 未知区域标记为0
    markers[unknown == 255] = 0

    # 分水岭算法分割
    markers = cv2.watershed(image, markers=markers)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(markers)
    markers_8u = np.uint8(markers)

    # 定义不同区域的颜色，用于后续可视化显示
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
              (255,0,255), (0,255,255), (255,128,0), (255,0,128),
              (128,255,0), (128,0,255), (255,128,128), (128,255,255)]
    # 遍历每个标记值，将每个区域的轮廓进行提取，并使用不同颜色对每个区域进行填充。同时，计算每个区域的中心，并在图像中进行标注。
    for i in range(2, int(max_val+1)):
        ret, thres1 = cv2.threshold(markers_8u, i-1, 255, cv2.THRESH_BINARY)
        ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
        mask = thres1 - thres2
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            cv2.drawContours(image, contours, -1, colors[(i-2)%12], -1)
    # # 遍历每个标记值，将每个区域的轮廓进行提取，并使用随机颜色对每个区域进行填充。同时，计算每个区域的中心，并在图像中进行标注。
    # for i in range(2, int(max_val+1)):
    #     ret, thres1 = cv2.threshold(markers_8u, i-1, 255, cv2.THRESH_BINARY)
    #     ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
    #     mask = thres1 - thres2
    #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     if len(contours) > 0:
    #         color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
    #         cv2.drawContours(image, contours, -1, color, -1)
            
    return image

image = cv2.imread('_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-17.tif')
src = image.copy()

# 边缘保留滤波EPF去噪，sp、sr分别表示空间窗口大小、色彩空间窗口大小
blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)

# 转成灰度图像
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# 使用otsu算法得到二值图像区间阈值
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 边缘检测
edges = cv2.Canny(binary, threshold1=30, threshold2=100)

# 边缘连接
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 颗粒分割
for contour in contours:
    # 可根据需要设置面积阈值来排除较小的区域
    # if cv2.contourArea(contour) > 100:
        # 创建一个与图像大小相同的空白掩码
        mask = np.zeros_like(image)
        # 使用当前轮廓绘制掩码
        cv2.drawContours(mask, [contour], 0, 255, -1)
        # 对掩码进行位操作，提取颗粒区域
        segmented_grain = cv2.bitwise_and(image, mask)

        # 在此可以对分割后的颗粒区域进行后处理

        # 显示分割结果
        cv2.imshow('Segmented Grain', segmented_grain)
        cv2.waitKey(0)

cv2.destroyAllWindows()





# # 读取待分割图像
# img = cv2.imread('_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-17.tif')

# # 图像预处理
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # 对每个区域进行处理
# for i, contour in enumerate(contours):
#     # 提取当前区域的边界框
#     x, y, w, h = cv2.boundingRect(contour)
#     # 对当前区域进行处理
#     segmented_roi = watershed_algorithm(img[y:y+h, x:x+w])
#     # 将处理结果覆盖到原图的对应区域上
#     img[y:y+h, x:x+w] = segmented_roi

# # 显示处理结果
# cv2.imshow('Result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()