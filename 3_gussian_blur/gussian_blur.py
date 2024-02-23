import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

def main():
    # 读取原始图像
    original_image = cv2.imread("panda.jpg")

    # 添加高斯噪声
    noisy_image = add_gaussian_noise(original_image)

    # 高斯滤波
    denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    # 显示高斯滤波的二维图像
    plt.subplot(131), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)), plt.title('Noisy Image')
    plt.subplot(133), plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)), plt.title('Denoised Image')

    # 保存对比图
    # plt.savefig("gaussian_filter_comparison.png")

    # 显示图像
    plt.show()

if __name__ == "__main__":
    main()
