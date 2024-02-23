from PIL import Image
# 打开彩色图像
color_image = Image.open('./cat.png')
# 转换为灰度图
gray_image = color_image.convert('L')
# 保存灰度图
gray_image.save('./gray_cat.jpg')

print("彩色图片格式: " + color_image.mode)
print("灰度图片格式: " + gray_image.mode)
