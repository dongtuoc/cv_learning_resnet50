import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

def gaussian_kernel(size, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(- ((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def plot_gaussian_kernel(kernel):
    fig = plt.figure()

    # 二维图像（像素值）
    ax1 = fig.add_subplot(121)
    ax1.imshow(kernel, cmap='viridis', interpolation='none')
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ax1.text(j, i, f'{kernel[i, j]:.2f}', ha='center', va='center', color='r')
    ax1.set_title('2D Gaussian Kernel')

    # 三维图像
    ax2 = fig.add_subplot(122, projection='3d')
    x = y = np.arange(0, kernel.shape[0], 1)
    x, y = np.meshgrid(x, y)
    ax2.plot_surface(x, y, kernel, cmap='viridis')
    ax2.set_title('3D Gaussian Kernel')

    plt.show()

def main():
    kernel_size = 5
    sigma = 1.0

    # 生成高斯滤波器
    kernel = gaussian_kernel(kernel_size, sigma)

    # 画出高斯滤波器的二维图像和三维图像
    plot_gaussian_kernel(kernel)

if __name__ == "__main__":
    main()
