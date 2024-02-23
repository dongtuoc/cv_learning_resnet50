# 本代码展示如何向图像中添加椒盐噪声，并使用均值滤波器进行噪声去除

# 导入必要的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数值计算
import matplotlib.pyplot as plt  # Matplotlib库，用于图像显示

# 使用OpenCV读取图像
image = cv2.imread("panda.png")  # 'panda.png'是图像文件的路径


# 这段代码定义了一个名为 add_salt_and_pepper_noise 的函数，用于向图像中添加椒盐噪声。
# 椒盐噪声是一种图像噪声，其中一些随机像素会被设置为最亮（通常是白色）或最暗（通常是黑色）
# 定义添加椒盐噪声的函数
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    向图像中添加椒盐噪声。

    参数:
    image (numpy.ndarray): 原始图像。
    salt_prob (float): 添加白色（盐）噪声的概率。
    pepper_prob (float): 添加黑色（椒）噪声的概率。

    返回:
    numpy.ndarray: 添加椒盐噪声后的图像。
    """
    noisy_image = image.copy()  # 创建图像的副本以避免修改原始图像
    total_pixels = image.size  # 计算图像中的总像素数

    # 添加椒盐噪声（黑色）
    num_salt = np.ceil(salt_prob * total_pixels)  # 根据椒盐噪声声概率计算需要添加的椒盐噪声声像素数
    salt_coords = [
        np.random.randint(0, i - 1, int(num_salt)) for i in image.shape
    ]  # 随机生成椒盐噪声的坐标
    noisy_image[salt_coords[0], salt_coords[1], :] = 255  # 将椒盐噪声像素设置为白色

    # 添加椒盐噪声（白色）
    num_pepper = np.ceil(pepper_prob * total_pixels)  # 根据椒盐噪声概率计算需要添加的椒盐噪声像素数
    pepper_coords = [
        np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape
    ]  # 随机生成椒盐噪声的坐标
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0  # 将椒盐噪声像素设置为黑色

    return noisy_image  # 返回添加了噪声的图像


# 向图像添加椒盐噪声
# 设置椒噪声和盐噪声的概率为 2%
salt_and_pepper_image = add_salt_and_pepper_noise(
    image, salt_prob=0.02, pepper_prob=0.02
)

# 定义均值滤波器的核大小
# 设置核大小为 8x8
kernel_size = (8, 8)

# 使用均值滤波器对图像进行滤波
# 应用均值滤波以减少图像中的噪声
filtered_image = cv2.blur(salt_and_pepper_image, kernel_size)

# 使用 Matplotlib 显示原始图像、添加椒盐噪声后的图像和滤波后的图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 设置绘图布局为 1 行 3 列

# 显示原始图像
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 将 BGR 格式转换为 RGB 格式
axes[0].set_title("Original Image")  # 设置图像标题
axes[0].axis("off")  # 关闭坐标轴显示

# 显示添加椒盐噪声后的图像
axes[1].imshow(cv2.cvtColor(salt_and_pepper_image, cv2.COLOR_BGR2RGB))  # 显示添加噪声后的图像
axes[1].set_title("Image with Salt and Pepper Noise")  # 设置图像标题
axes[1].axis("off")  # 关闭坐标轴显示

# 显示应用均值滤波后的图像
axes[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))  # 显示滤波后的图像
axes[2].set_title("Filtered Image")  # 设置图像标题
axes[2].axis("off")  # 关闭坐标轴显示

plt.show()  # 展示所有图像
