# 本代码展示如何使用 PIL, NumPy 和 matplotlib 处理并展示彩色 RGB 图像的不同颜色通道

# 导入必要的库
from PIL import Image  # 用于图像处理的PIL库
import numpy as np  # 用于数值计算的NumPy库
import matplotlib.pyplot as plt  # 用于显示图像的matplotlib.pyplot库

# 打开彩色图像文件
image = Image.open("./cat.jpg")  # './cat.jpg'是图像文件的路径

# 将PIL图像对象转换为NumPy数组
image_array = np.array(image)

# 图像的维度为 [high, width, channel]，[:,:,0] 代表是所有长宽以及通道0，也就是红色通道

# 分离RGB颜色通道中的红色通道
red_channel = image_array[:, :, 0]
# 分离RGB颜色通道中的绿色通道
green_channel = image_array[:, :, 1]
# 分离RGB颜色通道中的蓝色通道
blue_channel = image_array[:, :, 2]

# 使用matplotlib展示原始图像
plt.subplot(221), plt.imshow(image), plt.title("Original Image")
# 使用matplotlib展示蓝色通道图像，使用蓝色调的颜色映射
plt.subplot(222), plt.imshow(blue_channel, cmap="Blues"), plt.title("Blue Channel")
# 使用matplotlib展示绿色通道图像，使用绿色调的颜色映射
plt.subplot(223), plt.imshow(green_channel, cmap="Greens"), plt.title("Green Channel")
# 使用matplotlib展示红色通道图像，使用红色调的颜色映射
plt.subplot(224), plt.imshow(red_channel, cmap="Reds"), plt.title("Red Channel")
# 显示所有子图
plt.show()
