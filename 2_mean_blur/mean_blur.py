import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('panda.png')

# 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    total_pixels = image.size
    
    # 添加椒盐噪声
    num_salt = np.ceil(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255
    
    num_pepper = np.ceil(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    
    return noisy_image

# 生成椒盐噪声
salt_and_pepper_image = add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)

# 定义均值滤波器的大小
kernel_size = (8, 8)

# 应用均值滤波
filtered_image = cv2.blur(salt_and_pepper_image, kernel_size)

# 使用Matplotlib显示原始图像、椒盐噪声图像和滤波后的图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(salt_and_pepper_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Image with Salt and Pepper Noise')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[2].set_title('Filtered Image')
axes[2].axis('off')

plt.show()
