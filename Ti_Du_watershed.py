import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters
import cv2
from skimage.segmentation import watershed
 
# image =color.rgb2gray(data.camera())
image = cv2.imread('_Archive/2y-08WC-SDS1h-12PAH2h-wc30m-17.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声
 
#将梯度值低于10的作为开始标记点
markers = filters.rank.gradient(denoised, morphology.disk(5)) <10
markers = ndi.label(markers)[0]
 
gradient = filters.rank.gradient(denoised, morphology.disk(2)) #计算梯度
labels = watershed(gradient, markers, mask=image) #基于梯度的分水岭算法
 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes
 
ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(gradient, cmap=plt.cm.Spectral, interpolation='nearest')
ax1.set_title("Gradient")
ax2.imshow(markers, cmap=plt.cm.Spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
ax3.set_title("Segmented")
 
for ax in axes:
    ax.axis('off')
 
fig.tight_layout()
plt.show()