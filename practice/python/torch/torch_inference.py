# 这段代码使用了 PyTorch 和预训练的 ResNet50 模型来对图像进行分类。

# 它首先加载了预训练的 ResNet50 模型，然后对指定目录下的所有图片进行预处理，并使用模型进行预测。
# 最后，它输出每张图片预测的前五个最可能的类别

# 导入所需的库
import torch  # PyTorch库，用于深度学习
import heapq  # 堆队列算法库

# 加载预训练的ResNet50模型
resnet50 = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
resnet50.eval()  # 将模型设置为评估模式

import os  # 用于处理文件和目录的库

# 定义图片目录
pic_dir = "../../pics/"
# 获取图片目录下所有图片文件的路径
file_to_predict = [
    pic_dir + f for f in os.listdir(pic_dir) if os.path.isfile(pic_dir + f)
]

from PIL import Image  # 用于图像处理的PIL库
from torchvision import transforms  # 用于图像预处理的transforms库

# 对每个文件进行预测
for filename in file_to_predict:
    # 打开图像文件
    input_image = Image.open(filename)
    # 定义预处理步骤
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),  # 将图像大小调整为224x224
            # transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),  # 将图像转换为PyTorch张量
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
        ]
    )
    # 对图像进行预处理
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 创建一个mini-batch

    # 使用模型对图像进行预测
    output = resnet50(input_batch)

    # 获取前五个最高概率的预测结果
    res = list(output[0].detach().numpy())
    index = heapq.nlargest(5, range(len(res)), res.__getitem__)

    print("\npredict picture: " + filename)
    # 读取类别名称
    with open("../imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
        for i in range(5):
            print("         top " + str(i + 1) + ": " + categories[index[i]])
