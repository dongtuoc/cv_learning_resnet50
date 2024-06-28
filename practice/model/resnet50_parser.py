# 这段代码用于从一个预训练的 ResNet50 模型中提取并保存网络的参数。
# 代码的工作流程包括加载 ResNet50 模型，然后逐层提取卷积层（Convolutional Layers）、
# 批归一化层（Batch Normalization Layers）和全连接层（Fully Connected Layers）的参数
# 并将这些参数保存到文本文件中

import numpy as np  # 导入 NumPy 库
from torchvision import models  # 导入 torchvision 中的模型库
import torch  # 导入 PyTorch 库

# 加载预训练的 ResNet50 模型
resnet50 = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
resnet50.eval()
print(resnet50)

# 定义权重和参数保存的文件夹路径
dump_dir = "./resnet50_weight/"


# 保存卷积层参数的函数
def save_conv_param(data, file):
    """
    保存卷积层的参数到文本文件中。

    参数:
    data (Conv2d): PyTorch中的卷积层对象。
    file (str): 保存参数的文件名的基础部分。
    """
    # 获取卷积层的核大小、步长、填充、输入通道数和输出通道数
    kh = data.kernel_size[0]  # 核高度
    sh = data.stride[0]  # 步长
    pad_l = data.padding[0]  # 填充
    ci = data.in_channels  # 输入通道数
    co = data.out_channels  # 输出通道数

    # 将这些参数组合成一个列表
    l = [ci, co, kh, sh, pad_l]

    # 使用 numpy 将这些参数保存到文本文件
    # 文件名由基础文件名和 "_param.txt" 组合而成
    np.savetxt(dump_dir + file + str("_param.txt"), l)


# 定义函数保存批归一化层参数
def save_bn_param(data, file):
    """
    保存批归一化层的参数到文本文件中。

    参数:
    data (BatchNorm2d): PyTorch中的批归一化层对象。
    file (str): 保存参数的文件名的基础部分。
    """
    # 提取批归一化层的参数：epsilon (eps) 和动量 (momentum)
    eps = data.eps  # epsilon，用于数值稳定性
    momentum = data.momentum  # 动量，用于运行时均值和方差的更新

    # 将这些参数组合成一个列表
    l = [eps, momentum]

    # 使用 numpy 将这些参数保存到文本文件
    # 文件名由基础文件名和 "_param.txt" 组合而成
    np.savetxt(dump_dir + file + "_param.txt", l)


def save(data, file):
    """
    保存给定层的权重和偏置到文本文件中。

    参数:
    data: 层对象，可以是卷积层、批归一化层或全连接层。
    file (str): 保存权重和偏置的文件名的基础部分。
    """
    # 如果是卷积层
    if isinstance(data, type(resnet50.conv1)):
        # 保存卷积层的参数
        save_conv_param(data, file)
        # 转置权重矩阵以适应自定义计算需求
        w = np.array(data.weight.data.cpu().numpy())
        w = np.transpose(w, (0, 2, 3, 1))
        # 保存权重到文本文件
        np.savetxt(dump_dir + file + "_weight.txt", w.reshape(-1, 1))

    # 如果是批归一化层
    if isinstance(data, type(resnet50.bn1)):
        # 保存批归一化层的参数
        save_bn_param(data, file)
        # 保存运行时均值和方差
        m = np.array(data.running_mean.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_running_mean.txt", m.reshape(-1, 1))

        v = np.array(data.running_var.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_running_var.txt", v.reshape(-1, 1))

        # 保存偏置和权重
        b = np.array(data.bias.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_bias.txt", b.reshape(-1, 1))

        w = np.array(data.weight.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_weight.txt", w.reshape(-1, 1))

    # 如果是全连接层
    if isinstance(data, type(resnet50.fc)):
        # 打印权重矩阵的形状
        print(data.weight.shape)
        # 保存全连接层的偏置和权重
        bias = np.array(data.bias.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_bias.txt", bias.reshape(-1, 1))

        w = np.array(data.weight.data.cpu().numpy())
        np.savetxt(dump_dir + file + "_weight.txt", w.reshape(-1, 1))


# 这段代码定义了 save_bottle_neck 函数，它用于保存 ResNet50 网络中特定残差块（bottleneck）的所有层的权重和参数。
# 这个函数遍历每个残差块中的卷积层和批归一化层，并调用 save 函数来保存它们的权重和参数。
# 如果残差块包含下采样层（downsample），它还会保存这些层的权重和参数。
def save_bottle_neck(layer, layer_index):
    """
    保存指定残差块中的所有层的权重和参数。

    参数:
    layer: ResNet50 中的一个残差块。
    layer_index (int): 残差块的索引，用于文件命名。
    """
    bottle_neck_idx = 0  # 初始化残差块内部的索引

    # 为残差块创建基础文件名
    layer_name = "resnet50_layer" + str(layer_index) + "_bottleneck"

    # 遍历残差块中的每个子层
    for bottleNeck in layer:
        # 保存卷积层和批归一化层的权重和参数
        save(bottleNeck.conv1, layer_name + str(bottle_neck_idx) + "_conv1")
        save(bottleNeck.bn1, layer_name + str(bottle_neck_idx) + "_bn1")
        save(bottleNeck.conv2, layer_name + str(bottle_neck_idx) + "_conv2")
        save(bottleNeck.bn2, layer_name + str(bottle_neck_idx) + "_bn2")
        save(bottleNeck.conv3, layer_name + str(bottle_neck_idx) + "_conv3")
        save(bottleNeck.bn3, layer_name + str(bottle_neck_idx) + "_bn3")

        # 如果存在下采样层，也保存其权重和参数
        if bottleNeck.downsample:
            save(
                bottleNeck.downsample[0],
                layer_name + str(bottle_neck_idx) + "_downsample_conv2d",
            )
            save(
                bottleNeck.downsample[1],
                layer_name + str(bottle_neck_idx) + "_downsample_batchnorm",
            )

        # 更新残差块内部的索引
        bottle_neck_idx += 1


# 保存 ResNet50 模型的初始卷积层权重和参数
save(resnet50.conv1, "resnet50_conv1")
# 保存 ResNet50 模型的初始批归一化层权重和参数
save(resnet50.bn1, "resnet50_bn1")

# 遍历并保存 ResNet50 模型中的四个残差块的权重和参数
# 保存第一残差块
save_bottle_neck(resnet50.layer1, 1)
# 保存第二残差块
save_bottle_neck(resnet50.layer2, 2)
# 保存第三残差块
save_bottle_neck(resnet50.layer3, 3)
# 保存第四残差块
save_bottle_neck(resnet50.layer4, 4)

# 保存 ResNet50 模型的全连接层权重和参数
save(resnet50.fc, "resnet50_fc")
