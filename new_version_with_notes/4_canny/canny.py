# 这段代码演示了如何使用 OpenCV 进行边缘检测。
# 它首先读取一幅图像，然后应用 Canny 算子来检测图像中的边缘。
# 最后，使用 Matplotlib 显示原始图像和检测到的边缘。

# 导入所需的库
import cv2  # OpenCV库，用于图像处理
import matplotlib.pyplot as plt  # Matplotlib库，用于显示图像

# 读取图像
# 'cat.png'是图像文件的路径，cv2.IMREAD_GRAYSCALE表示以灰度模式读取图像
image = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE)

# 使用Canny边缘检测算法检测图像边缘
# cv2.Canny函数接收图像和两个阈值（低阈值和高阈值），这里设置为50和150
edges = cv2.Canny(image, 50, 150)

# 使用Matplotlib显示结果
plt.figure(figsize=(8, 4))  # 设置图像显示大小

# 显示原始图像
plt.subplot(1, 2, 1)  # 创建一个1行2列的子图，并定位到第一个
plt.imshow(image, cmap="gray")  # 以灰度图模式显示原始图像
plt.title("Original Image")  # 设置子图的标题

# 显示使用Canny算法检测到的边缘
plt.subplot(1, 2, 2)  # 定位到第二个子图
plt.imshow(edges, cmap="gray")  # 以灰度图模式显示边缘图像
plt.title("Canny Edges")  # 设置子图的标题

plt.show()  # 显示所有子图
