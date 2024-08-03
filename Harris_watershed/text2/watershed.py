# 打开一个新文件，将其命名为 watershed.py ，然后插入以下代码：

# 导入必要的包
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2

# 构造参数解析并解析参数
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", default="HFOUG.jpg", help="path to input image")
ap.add_argument("-i", "--image", default="watershed_coins_01.jpg", help="path to input image")
args = vars(ap.parse_args())
# 加载图像并执行金字塔均值偏移滤波以辅助阈值化步骤
image = cv2.imread(args["image"])
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.imshow("Input", image)
# 将图像转换为灰度，然后应用大津阈值
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

# 计算从每个二进制图像中的像素到最近的零像素的精确欧氏距离，然后找出这个距离图中的峰值
D = ndimage.distance_transform_edt(thresh)

# 可视化距离函数
D_show = cv2.normalize(D, None, 0, 1, cv2.NORM_MINMAX)
# print(np.max(D_show))
cv2.imshow("D_show", D_show)


# 以坐标列表(indices=True)或布尔掩码(indices=False)的形式查找图像中的峰值。峰值是2 * min_distance + 1区域内的局部最大值。
# (即峰值之间至少相隔min_distance)。此处我们将确保峰值之间至少有20像素的距离。
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
# 可视化localMax
temp = localMax.astype(np.uint8)
cv2.imshow("localMax", temp * 255)
# 使用8-连通性对局部峰值进行连接成分分析，然后应用分水岭算法
# scipy.ndimage.label(input, structure=None, output=None)
# input ：待标记的数组对象。输入中的任何非零值都被视为待标记对象，零值被视为背景。
# structure：定义要素连接的结构化元素。对于二维数组。默认是四连通， 此处选择8连通
#
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 可视化markers
temp_markers = markers.astype(np.uint8)
cv2.imshow("temp_markers", temp_markers * 20)

# 由于分水岭算法假设我们的标记代表距离图中的局部最小值（即山谷），因此我们取 D 的负值。
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# 循环遍历分水岭算法返回的标签
for label in np.unique(labels):
    # 0表示背景，忽略它
    if label == 0:
        continue
    # 否则，为标签区域分配内存并将其绘制在掩码上
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    # 在mask中检测轮廓并抓取最大的一个
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # 在物体周围画一个圆
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# 显示输出图像
cv2.imshow("Output", image)
cv2.waitKey(0)
