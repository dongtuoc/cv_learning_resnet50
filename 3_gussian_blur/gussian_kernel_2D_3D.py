# 这段代码展示了如何生成一个高斯滤波器（高斯核），并使用 matplotlib 绘制其二维和三维图像。
# 高斯滤波器是一种常用于图像处理中的平滑滤波器，它根据高斯函数生成一个核，该核在图像处理中用于模糊图像和去除噪声。

# 导入所需的库
import numpy as np  # NumPy库，用于数值计算
import matplotlib.pyplot as plt  # Matplotlib库，用于图像显示
from mpl_toolkits.mplot3d import Axes3D  # Matplotlib的3D绘图工具包
from scipy.stats import multivariate_normal  # SciPy的多元正态分布函数


# 定义生成高斯核的函数
def gaussian_kernel(size, sigma=1.0):
    """
    生成高斯核。

    参数:
    size (int): 高斯核的大小。
    sigma (float): 高斯核的标准差。

    返回:
    numpy.ndarray: 高斯核。
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma**2)),
        (size, size),
    )
    return kernel / np.sum(kernel)  # 归一化核


# 定义绘制高斯核的函数
def plot_gaussian_kernel(kernel):
    """
    绘制和可视化高斯核的二维和三维图像。

    参数:
    kernel (numpy.ndarray): 要绘制的高斯核。
    """
    fig = plt.figure()  # 创建绘图对象

    # 绘制二维高斯核图像
    ax1 = fig.add_subplot(121)  # 添加子图位于左侧
    ax1.imshow(kernel, cmap="viridis", interpolation="none")  # 显示高斯核，使用viridis颜色映射
    # 在图像上标注每个像素的值
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ax1.text(j, i, f"{kernel[i, j]:.2f}", ha="center", va="center", color="r")
    ax1.set_title("2D Gaussian Kernel")  # 设置图像标题

    # 绘制三维高斯核图像
    ax2 = fig.add_subplot(122, projection="3d")  # 添加子图位于右侧，并设置为3D模式
    x, y = np.arange(0, kernel.shape[0], 1), np.arange(
        0, kernel.shape[1], 1
    )  # 创建x和y坐标网格
    x, y = np.meshgrid(x, y)
    ax2.plot_surface(x, y, kernel, cmap="viridis")  # 绘制高斯核的3D表面图
    ax2.set_title("3D Gaussian Kernel")  # 设置图像标题

    plt.show()  # 展示绘制的图像


kernel_size = 5  # 定义核的大小
sigma = 1.0  # 定义高斯核的标准差

# 生成高斯滤波器
kernel = gaussian_kernel(kernel_size, sigma)

# 绘制高斯滤波器的二维图像和三维图像
plot_gaussian_kernel(kernel)
