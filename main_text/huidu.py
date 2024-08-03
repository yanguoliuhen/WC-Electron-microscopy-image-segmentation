# import cv2
# import numpy as np


# def local_maxima(image):
#     # 距离变换
#     dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)

#     # 局部最大值滤波器
#     kernel = np.ones((3, 3), np.uint8)
#     local_max_filter = cv2.dilate(dist_transform, kernel, iterations=2)

#     # 找到局部最大值
#     local_maxima = np.zeros_like(image)
#     local_maxima[np.where(local_max_filter == dist_transform)] = 255

#     return local_maxima


# # 读取图像
# image = cv2.imread('_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-10.png', 0)

# # 转换为二值图像（可根据需要进行阈值处理）
# ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # 获取局部最大值图像
# local_maxima_image = local_maxima(binary)
# print(local_maxima_image)
# # 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Binary Image', binary)
# cv2.imshow('Local Maxima Image', local_maxima_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import numpy as np
from scipy.ndimage import grey_dilation
from skimage import io, img_as_ubyte
import cv2
import numpy as np


def h_maxima_transform(image, h):
    #     # Convert the image to unsigned byte format
    #     image = img_as_ubyte(image)
    # # 边缘保留滤波EPF去噪，sp、sr分别表示空间窗口大小、色彩空间窗口大小
    blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)

    # 转成灰度图像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 使用otsu算法得到二值图像区间阈值
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    # image = img_as_ubyte(dist_out)

    # Find the maximum value in the image
    f = np.max(dist_out)

    # Calculate the threshold value
    threshold = f - h

    # Create a mask of pixels greater than the threshold
    mask = image > threshold

    # Perform dilation and reconstruction using the mask
    dilated = grey_dilation(image, size=(3, 3))
    reconstructed = (dilated * mask).clip(0, f)

    return reconstructed


# Load the grayscale image
image = io.imread(
    '_Archive_noLogo/2y-08WC-SDS1h-12PAH2h-wc30m-10.png')

# Perform h-maxima transformation with h = 10
h_maxima = h_maxima_transform(image, h=20)

# Display the resulting image
io.imshow(h_maxima)
io.show()
