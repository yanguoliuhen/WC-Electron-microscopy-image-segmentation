import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def preprocess_image(img_path, threshold_value=120):
    # 读取图像
    img = cv2.imread(img_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # 自适应高斯阈值二值化
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)

    # 使用固定阈值将灰度图像进行二值化
    # _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 使用 Otsu's 二值化
    thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 如果待测粒子为黑色背景为白色，则对图像进行反转
    binary = cv2.bitwise_not(binary)

    return binary

def edge_detection(binary):
    # 使用高斯拉普拉斯边缘检测获得边缘信息
    edges = cv2.Laplacian(binary, cv2.CV_8U, ksize=3)

    return edges

def detect_keypoints(edges):
    # 使用FAST算法遍历边缘信息获得边缘特征点
    keypoints = cv2.FastFeatureDetector_create().detect(edges)

    return keypoints

def plot_keypoints(image, keypoints):
    # 绘制特征点
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    return img_with_keypoints

def detect_edge_features(edges):

    # 使用FAST算法获取边缘特征点
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(edges, None)

    return keypoints

def distance_transform(binary):
    # 计算距离变换
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 归一化距离变换
    norm_dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

    # 将距离变换转换为灰度图像
    dist_transform_gray = cv2.convertScaleAbs(norm_dist_transform)

    return dist_transform_gray

def watershed_lines(dist_transform_gray, binary, h_value=0.5):
    # 应用 h-maxima 转换
    h = 0.5
    h_max = cv2.dilate(dist_transform_gray, np.ones((3,3))) - dist_transform_gray
    h_max[h_max < h] = 0

    # 应用分水岭变换
    ret, markers = cv2.connectedComponents(np.uint8(h_max))
    markers += 1
    markers[dist_transform_gray == 0] = 0
    markers = cv2.watershed(cv2.cvtColor(dist_transform_gray, cv2.COLOR_GRAY2BGR), markers)

    # 生成分水岭脊线
    ridge_lines = np.zeros_like(dist_transform_gray, np.uint8)
    ridge_lines[markers == -1] = 255

    return ridge_lines

if __name__ == '__main__':
    # 导入图像
    img_path = '_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-17.tif'
    image = cv2.imread(img_path)

    # 1.对图像进行预处理
    binary = preprocess_image(img_path)
    
    # 2.高斯拉普拉斯边缘检测获取边缘信息
    edges = edge_detection(binary)

    # 3.使用FAST算法遍历edges，获得边缘特征点
    keypoints = detect_keypoints(edges)

    img_with_keypoints = plot_keypoints(image, keypoints)

    # 2.进行欧式距离变换
    dist_transform_gray = distance_transform(binary)

    # 2.应用分水岭算法绘制分水岭脊线或候选分割线
    ridge_lines = watershed_lines(dist_transform_gray, binary, h_value=0.5)

    # 原始图像与脊线图像叠加
    result = cv2.addWeighted(image, 0.7, cv2.cvtColor(ridge_lines, cv2.COLOR_GRAY2BGR), 0.3, 0)

#--------------------------------------------------------------------------------结果绘制区

    # # 绘制二值图像
    plt.imshow(binary, cmap='gray')
    plt.show()

    # 进行高斯拉普拉斯边缘检测并绘制结果图
    plt.imshow(edges, cmap='gray')
    plt.show()

    # 显示FAST后边缘特征点结果图像
    plt.imshow(img_with_keypoints)
    plt.show()

    # 显示欧式距离变换图像
    plt.imshow(dist_transform_gray, cmap='gray')
    plt.show()

    # 显示分水岭分割结果图像
    plt.imshow(ridge_lines, cmap='nipy_spectral')
    plt.show()

    # # 原始图像与脊线图像叠加
    # plt.imshow(result,cmap='nipy_spectral')
    # plt.show()

    # 绘制分水岭脊线图像
    plt.imshow(cv2.cvtColor(ridge_lines, cv2.COLOR_BGR2RGB))
    plt.show()








