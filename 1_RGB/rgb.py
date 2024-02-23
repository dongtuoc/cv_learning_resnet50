from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 打开彩色图像
image = Image.open('./cat.png')

# 将图像转换为 NumPy 数组
image_array = np.array(image)

# 分离通道
red_channel = image_array[:, :, 0]
green_channel = image_array[:, :, 1]
blue_channel = image_array[:, :, 2]

# 显示原始图像和各个通道
plt.subplot(221), plt.imshow(image), plt.title('Original Image')
plt.subplot(222), plt.imshow(blue_channel, cmap='Blues'), plt.title('Blue Channel')
plt.subplot(223), plt.imshow(green_channel, cmap='Greens'), plt.title('Green Channel')
plt.subplot(224), plt.imshow(red_channel, cmap='Reds'), plt.title('Red Channel')
plt.show()
