import cv2
import numpy as np
import random as rd

def watershed_algorithm(image):
    src = image.copy()

    # 边缘保留滤波EPF去噪，sp、sr分别表示空间窗口大小、色彩空间窗口大小
    blur = cv2.pyrMeanShiftFiltering(image,sp=21,sr=55)
    # cv2.imshow("blur", blur)

    # 转成灰度图像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 使用otsu算法得到二值图像区间阈值
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('thres image', binary)

    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow('distance-Transform', dist_out * 100)

    # 对距离变换图像进行二值化处理，得到前景区域的初始标记，0.5*dist_out.max()表示阈值
    ret, surface = cv2.threshold(dist_out, 0.8*dist_out.max(), 255, cv2.THRESH_BINARY)
    # cv2.imshow('surface', surface)

    # 转成8位整型
    sure_fg = np.uint8(surface)
    # cv2.imshow('Sure foreground', sure_fg)

    # 使用连通组件分析算法，得到初始标记图像，其中前景区域的标记值为2，背景区域的标记值为1
    ret, markers = cv2.connectedComponents(sure_fg)  # 连通区域
    print(ret)
    markers = markers + 1 #整个图+1，使背景不是0而是1值

    # 对二值图像进行膨胀处理，得到未知区域的标记。将未知区域的标记值设为0
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    unknown = binary - sure_fg
    # cv2.imshow('unknown',unknown)

    # 未知区域标记为0
    markers[unknown == 255] = 0
    # 区域标记结果
    # markers_show = np.uint8(markers)
    # cv2.imshow('markers',markers_show*100)

    # 分水岭算法分割
    markers = cv2.watershed(image, markers=markers)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(markers)
    markers_8u = np.uint8(markers)

    # 定义不同区域的颜色，用于后续可视化显示
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
              (255,0,255), (0,255,255), (255,128,0), (255,0,128),
              (128,255,0), (128,0,255), (255,128,128), (128,255,255)]
    # 遍历每个标记值，将每个区域的轮廓进行提取，并使用不同颜色对每个区域进行填充。同时，计算每个区域的中心，并在图像中进行标注。
    for i in range(2,int(max_val+1)):
        ret, thres1 = cv2.threshold(markers_8u, i-1, 255, cv2.THRESH_BINARY)
        ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
        mask = thres1 - thres2
        # cv2.imshow('mask',mask)
        
        contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image,contours,-1,colors[(i-2)%12],-1)
        # cv2.drawContours(src,contours,-1,colors[(i-2)%12],-1)
    
    # 在原图和分割后的图像中，添加区域数量的文字标注。
    # cv2.putText(src,"count=%d"%(int(max_val-1)),(220,30),0,1,(0,255,0),2)
    # cv2.putText(image,"count=%d"%(int(max_val-1)),(220,30),0,1,(0,255,0),2)

    # 将原图和分割后的图像进行叠加，得到可视化显示的结果
    # cv2.imshow('regions', image)
    result = cv2.addWeighted(src,0.6,image,0.5,0) #图像权重叠加
    cv2.imshow('result', result)

src = cv2.imread('2y-08WC-SDS1h-12PAH2h-wc30m-07.png')
# cv2.imshow('src', src)
watershed_algorithm(src)
cv2.waitKey(0)
cv2.destroyAllWindows()