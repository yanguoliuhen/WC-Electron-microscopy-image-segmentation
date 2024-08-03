import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology, segmentation, color, io

# 读取图像
rgb = cv2.imread("./2y-08WC-SDS1h-12PAH2h-wc30m-07.png")

# 转换为灰度图像
I = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# 显示原始灰度图像
plt.imshow(I, cmap='gray')
plt.show()

# 显示图像来源信息
plt.text(732, 501, "Image courtesy of Corel(R)", fontsize=7, horizontalalignment="right")

# 计算图像梯度
gmag = cv2.Sobel(I, cv2.CV_64F, 1, 1, ksize=3)
plt.imshow(np.abs(gmag), cmap='gray')
plt.title("Gradient Magnitude")
plt.show()

# 定义结构元素
se = morphology.disk(5)

# 图像开运算
Io = morphology.opening(I, se)
plt.imshow(Io, cmap='gray')
plt.title("Opening")
plt.show()

# 图像腐蚀
Ie = morphology.erosion(I, se)
Iobr = morphology.reconstruction(Ie, I)
plt.imshow(Iobr, cmap='gray')
plt.title("Opening-by-Reconstruction")
plt.show()

# 图像闭运算
Ioc = morphology.closing(Io, se)
plt.imshow(Ioc, cmap='gray')
plt.title("Opening-Closing")
plt.show()

# 图像开闭运算重建
Iobrd = morphology.dilation(Iobr, se)
Iobrcbr = morphology.reconstruction(Iobrd, Iobr)
plt.imshow(Iobrcbr, cmap='gray')
plt.title("Opening-Closing by Reconstruction")
plt.show()

# 寻找区域最大值
fgm = morphology.regional_max(Iobrcbr)
plt.imshow(fgm, cmap='gray')
plt.title("Regional Maxima of Opening-Closing by Reconstruction")
plt.show()

# 在原始图像上叠加区域最大值
I2 = color.label2rgb(I, fgm)
plt.imshow(I2)
plt.title("Regional Maxima Superimposed on Original Image")
plt.show()

# 定义新的结构元素
se2 = morphology.square(5)

# 图像闭运算和腐蚀
fgm2 = morphology.closing(fgm, se2)
fgm3 = morphology.erosion(fgm2, se2)

# 移除小区域
fgm4 = morphology.remove_small_objects(fgm3, min_size=20)

# 在原始图像上叠加修改后的区域最大值
I3 = color.label2rgb(I, fgm4)
plt.imshow(I3)
plt.title("Modified Regional Maxima Superimposed on Original Image")
plt.show()

# 图像二值化
ret, bw = cv2.threshold(
        Iobrcbr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 计算距离变换
D = morphology.distance_transform_edt(bw)

# 分水岭算法
DL = segmentation.watershed(D)
bgm = DL == 0

plt.imshow(bgm, cmap='gray')
plt.title("Watershed Ridge Lines")
plt.show()

# 进一步处理得到标记
fgm4 = morphology.remove_small_objects(fgm3, min_size=20)
I4 = color.label2rgb(I, fgm4)

# 显示标记和对象边界
plt.imshow(I4)
plt.title("Markers and Object Boundaries Superimposed on Original Image")
plt.show()

# 显示彩色分水岭标签矩阵
Lrgb = color.label2rgb(DL, bg_label=0)
plt.imshow(Lrgb)
plt.title("Colored Watershed Label Matrix")
plt.show()

# 在原始图像上叠加彩色标签
plt.imshow(I)
plt.hold(True)
himage = plt.imshow(Lrgb, alpha=0.3)
plt.title("Colored Labels Superimposed Transparently on Original Image")
plt.show()


