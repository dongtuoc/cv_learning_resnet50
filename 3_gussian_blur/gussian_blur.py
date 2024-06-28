# 这段代码展示了如何向图像中添加高斯噪声，并使用高斯滤波进行噪声去除。
# 代码首先读取原始图像，然后添加高斯噪声，并应用高斯滤波。
# 最后，使用 matplotlib 展示原始图像、添加噪声后的图像和去噪后的图像。

# 导入所需的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数值计算
import matplotlib.pyplot as plt  # Matplotlib库，用于显示图像


# 这段代码定义了一个名为 add_gaussian_noise 的函数，它用于向图像中添加高斯噪声。
# 高斯噪声是一种常见的图像噪声，它对图像的每个像素添加了根据高斯分布生成的随机值。
def add_gaussian_noise(image, mean=0, sigma=25):
    """
    向图像中添加高斯噪声。

    参数:
    image (numpy.ndarray): 原始图像。
    mean (float): 高斯噪声的均值，默认为 0。
    sigma (float): 高斯噪声的标准差，默认为 25。

    返回:
    numpy.ndarray: 添加高斯噪声后的图像。
    """
    # 获取图像的维度和通道数
    row, col, ch = image.shape

    # 生成与图像尺寸相同的高斯噪声
    gauss = np.random.normal(mean, sigma, (row, col, ch))

    # 将高斯噪声添加到原始图像上
    # 使用 np.clip 确保添加噪声后的像素值仍然在 0 到 255 的范围内
    noisy = np.clip(image + gauss, 0, 255)

    # 返回添加噪声后的图像，并将数据类型转换为无符号8位整型
    return noisy.astype(np.uint8)


# 读取图像文件
original_image = cv2.imread("panda.jpg")  # 'panda.jpg' 是图像文件的路径

# 向原始图像添加高斯噪声
noisy_image = add_gaussian_noise(original_image)

# 使用高斯滤波器对添加噪声的图像进行去噪处理
# 设置高斯滤波器的核大小为 5x5
denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# 使用 matplotlib 显示原始图像、带噪声的图像和去噪后的图像
plt.subplot(131)  # 设置显示位置
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # 显示原始图像（转换颜色空间为RGB）
plt.title("Original Image")  # 设置图像标题

plt.subplot(132)  # 设置显示位置
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))  # 显示带噪声的图像
plt.title("Noisy Image")  # 设置图像标题

plt.subplot(133)  # 设置显示位置
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))  # 显示去噪后的图像
plt.title("Denoised Image")  # 设置图像标题

# 显示所有图像
plt.show()
