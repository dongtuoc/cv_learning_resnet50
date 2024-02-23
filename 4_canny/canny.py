import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

# 使用 Canny 算子进行边缘检测
edges = cv2.Canny(image, 50, 150)  # 调整阈值以获得最佳效果

# 显示结果
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')

plt.show()
