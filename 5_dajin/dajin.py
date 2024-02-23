import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('luna.jfif', cv2.IMREAD_GRAYSCALE)

# 大津算法找到最佳阈值
_, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 使用 Matplotlib 显示原始图像和分割结果
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(thresholded, cmap='gray')
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])

plt.show()

