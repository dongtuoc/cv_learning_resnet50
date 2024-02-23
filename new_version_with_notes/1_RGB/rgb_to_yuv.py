# 这段代码用于演示如何将RGB图像转换为YUV格式，并展示其Y、U、V各通道的效果

import cv2  # 导入OpenCV库，用于图像处理
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot，用于图像显示

# 使用OpenCV的imread函数读取指定路径的图像
rgb_image = cv2.imread("./cat.png")  # './cat.png'是图像的路径，需要根据实际情况修改

# 将读取的RGB图像转换为YUV格式
# cv2.cvtColor函数用于转换图像的颜色空间，这里从BGR转换为YUV
# OpenCV中默认读取的格式为BGR，而不是RGB
yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)

# 使用cv2.split函数分离YUV图像的三个通道（Y, U, V）
y_channel, u_channel, v_channel = cv2.split(yuv_image)

# 使用matplotlib的subplot函数和imshow函数显示四个子图
# 第一个子图为原始的RGB图像
plt.subplot(221), plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)), plt.title(
    "Original RGB feature"
)
# 第二个子图为Y通道的图像，使用灰度图显示
plt.subplot(222), plt.imshow(y_channel, cmap="gray"), plt.title("Y channel")
# 第三个子图为U通道的图像，同样使用灰度图显示
plt.subplot(223), plt.imshow(u_channel, cmap="gray"), plt.title("U channel")
# 第四个子图为V通道的图像
plt.subplot(224), plt.imshow(v_channel, cmap="gray"), plt.title("V channel")
# 显示所有子图
plt.show()
