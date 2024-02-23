import cv2
import matplotlib.pyplot as plt

# 读取 RGB 图像
rgb_image = cv2.imread('./cat.png')

# 将 RGB 图像转换为 YUV 格式
yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)

# 分离 Y、U、V 通道
y_channel, u_channel, v_channel = cv2.split(yuv_image)

# 显示原始 RGB 图像和 Y、U、V 通道
plt.subplot(221), plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)), plt.title('Original RGB Image')
plt.subplot(222), plt.imshow(y_channel, cmap='gray'), plt.title('Y Channel')
plt.subplot(223), plt.imshow(u_channel, cmap='gray'), plt.title('U Channel')
plt.subplot(224), plt.imshow(v_channel, cmap='gray'), plt.title('V Channel')
plt.show()
