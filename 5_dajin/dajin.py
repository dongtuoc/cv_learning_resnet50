# 这段代码展示了如何使用 OpenCV 库进行图像分割。
# 它首先读取一幅灰度图像，然后使用大津算法（Otsu's method）自动找到一个阈值来将图像分割为二值图像（黑白图像）。
# 最后，使用 Matplotlib 将原始图像和分割后的图像显示出来。

# 导入所需的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数值计算
import matplotlib.pyplot as plt  # Matplotlib库，用于显示图像

# 读取图像
# 'luna.jfif'是图像文件的路径，cv2.IMREAD_GRAYSCALE表示以灰度模式读取图像
image = cv2.imread("luna.jfif", cv2.IMREAD_GRAYSCALE)

# 使用大津算法（Otsu's method）自动找到最佳阈值进行图像二值化
# cv2.threshold函数返回两个值，第一个是找到的阈值，第二个是阈值化后的图像
# cv2.THRESH_BINARY是二值化类型，cv2.THRESH_OTSU是使用Otsu算法
_, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 使用Matplotlib显示原始图像和分割后的图像
# 显示原始图像
plt.subplot(121), plt.imshow(image, cmap="gray")  # 在1行2列的子图中的第一个位置显示原始图像
plt.title("Original Image"), plt.xticks([]), plt.yticks([])  # 设置标题和去除坐标轴

# 显示分割后的图像
plt.subplot(122), plt.imshow(thresholded, cmap="gray")  # 在第二个位置显示分割后的图像
plt.title("Segmented Image"), plt.xticks([]), plt.yticks([])  # 设置标题和去除坐标轴

plt.show()  # 显示所有子图
