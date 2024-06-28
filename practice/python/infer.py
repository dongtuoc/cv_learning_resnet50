# 这段代码是一个自定义实现的图像分类程序。
# 它使用了类似于 ResNet50 架构的卷积神经网络来对图片进行分类。
# 代码首先读取图像，然后通过一系列层（卷积、批归一化、ReLU激活、池化和全连接）处理图像，并输出分类结果。

import numpy as np

# 设置模型参数文件的前缀，可以打开 ../model/resnet_weight/ 目录下查看参数文件的命名。
# 如 resnet50_bn1_bias.txt 为第一个bn层的 bias 参数。
# 这里先定义一个文件前缀，后面会拼接出一个完整的文件名。
params_dir = "../model/resnet50_weight/resnet50_"


# 如果运行程序耗时时间长，可以将use_opt 设置为 True，使能 np.dot 来替代卷积的乘累加运算，加速运算性能
use_opt = True


def LoadDataFromFile(file_name, is_float=True):
    """
    从文件中加载数据。

    参数:
    file_name (str): 要读取的文件的路径。
    is_float (bool): 指示加载的数据是否应被解释为浮点数。默认为 True。
                     如果为 False，则数据将被解释为整数。

    返回:
    list: 包含文件中数据的列表。
    """
    # 定义一个空列表用于存放读取的数据
    k = []
    # 打开指定的文件进行读取
    with open(file_name, "r") as f_:
        # 读取文件的所有行
        lines = f_.readlines()
        # 将每行转换为浮点数并存储在列表中
        k = [float(l) for l in lines]
        # 如果指定为非浮点数，则将列表中的元素转换为整数
        if not is_float:
            k = [int(l) for l in k]
    # 返回包含数据的列表
    return k


# 以下一些函数，调用LoadDataFromFile，并且拼接出完整文件名字


# 加载卷积权值的函数
def LoadConvWeight(name):
    """
    加载指定卷积层的权重。

    参数:
    name (str): 卷积层的名称，用于确定要读取的权重文件。

    返回:
    list: 包含卷积权重的列表。
    """
    # 构造权重文件的完整路径
    name = params_dir + name + "_weight.txt"
    # 可以取消下面这行的注释来打印出正在读取的权重文件名
    # print(name)
    # 调用 LoadDataFromFile 函数读取权重数据
    return LoadDataFromFile(name, is_float=True)


# 加载卷积参数的函数
def LoadConvParam(name):
    """
    加载指定卷积层的参数。

    参数:
    name (str): 卷积层的名称，用于确定要读取的参数文件。

    返回:
    list: 包含卷积参数的列表。
    """
    # 构造参数文件的完整路径
    name = params_dir + name + "_param.txt"
    # 调用 LoadDataFromFile 函数读取参数数据
    param = LoadDataFromFile(name, is_float=False)
    return param


# 导入卷积计算的接口，卷积计算的接口代码实现在 python/ops/conv2d 目录下
import ops.conv2d as conv


def ComputeConvLayer(in_data, layer_name):
    """
    计算指定卷积层的输出。

    参数:
    in_data (numpy.ndarray): 输入数据，维度为 [h, w, c]，其中 h、w 和 c 分别表示高度、宽度和通道数。
    layer_name (str): 卷积层的名称，用于从预存的文件中加载相应的权重和参数。

    返回:
    numpy.ndarray: 卷积操作后的输出数据。
    """
    print("-- compute " + layer_name)
    # 加载卷积层的权重
    weight = LoadConvWeight(layer_name)
    # 加载卷积层的参数
    param = LoadConvParam(layer_name)
    # 解析卷积层参数
    hi, wi = in_data.shape[0], in_data.shape[1]  # 输入数据的高度和宽度
    ci, co, kernel, stride, pad = param  # 分别为输入通道数、输出通道数、核大小、步长和填充

    # 根据是否使用优化方法（use_opt）选择不同的卷积操作
    if use_opt:
        # 使用优化的卷积操作
        res = conv.Conv2dOpt(in_data, weight, hi, wi, ci, co, kernel, stride, pad)
    else:
        # 使用普通的卷积操作
        res = conv.Conv2d(in_data, weight, hi, wi, ci, co, kernel, stride, pad)
    print(res.shape)
    return res


# 导入全连接计算的接口，卷积计算的接口代码实现在 python/ops/fc 目录下
import ops.fc as fc


def ComputeFcLayer(in_data, layer_name):
    """
    计算指定全连接层的输出。

    参数:
    in_data (numpy.ndarray): 全连接层的输入数据。
    layer_name (str): 全连接层的名称，用于确定加载权重和偏置的文件。

    返回:
    numpy.ndarray: 全连接层的输出数据。
    """
    print("-- compute " + layer_name)
    # 构造权重和偏置文件的完整路径
    weight_file_name = params_dir + layer_name + "_weight.txt"
    bias_file_name = params_dir + layer_name + "_bias.txt"
    # 加载权重和偏置
    weight = LoadDataFromFile(weight_file_name)
    bias = LoadDataFromFile(bias_file_name)
    # 根据是否使用优化方法选择不同的全连接计算
    if use_opt:
        # 使用优化的全连接计算
        res = fc.FullyConnectOpt(in_data, weight, bias)
    else:
        # 使用标准的全连接计算
        res = fc.FullyConnect(in_data, weight, bias)
    print(res.shape)
    return res


# 导入bn计算的接口，卷积计算的接口代码实现在 python/ops/bn 目录下
import ops.bn as bn


# 这段代码定义了 ComputeBatchNormLayer 函数，用于计算批归一化（Batch Normalization）层的输出。
# 该函数首先从指定的文件中加载层的权重、偏置、均值和方差，然后应用批归一化公式来标准化输入数据
def ComputeBatchNormLayer(in_data, layer_name):
    """
    计算指定批归一化层的输出。

    参数:
    in_data (numpy.ndarray): 批归一化层的输入数据。
    layer_name (str): 批归一化层的名称，用于确定加载相关参数的文件。

    返回:
    numpy.ndarray: 批归一化层的输出数据。
    """
    print("-- compute " + layer_name)
    # 构造相关参数文件的完整路径
    weight_file_name = params_dir + layer_name + "_weight.txt"
    bias_file_name = params_dir + layer_name + "_bias.txt"
    mean_file_name = params_dir + layer_name + "_running_mean.txt"
    var_file_name = params_dir + layer_name + "_running_var.txt"
    # 加载批归一化所需的参数：权重、偏置、均值和方差
    weight = LoadDataFromFile(weight_file_name)
    bias = LoadDataFromFile(bias_file_name)
    mean = LoadDataFromFile(mean_file_name)
    var = LoadDataFromFile(var_file_name)
    # 应用批归一化公式处理输入数据
    res = bn.BatchNorm(in_data, mean, var, weight, bias)
    print(res.shape)
    return res


# 导入池化计算的接口，卷积计算的接口代码实现在 python/ops/pool 目录下
import ops.pool as pool


def ComputeMaxPoolLayer(in_data):
    """
    执行最大池化操作。

    参数:
    in_data (numpy.ndarray): 需要进行最大池化的输入数据。

    返回:
    numpy.ndarray: 最大池化后的结果。
    """
    print("-- compute maxpool")
    # 调用最大池化函数处理输入数据
    res = pool.MaxPool(in_data)
    print(res.shape)
    return res


def ComputeAvgPoolLayer(in_data):
    """
    执行平均池化操作。

    参数:
    in_data (numpy.ndarray): 需要进行平均池化的输入数据。

    返回:
    numpy.ndarray: 平均池化后的结果。
    """
    print("-- compute avgpool")
    # 调用平均池化函数处理输入数据
    res = pool.AvgPool(in_data)
    print(res.shape)
    return res


# 这段代码定义了一个名为 ComputeReluLayer 的函数
# 用于计算神经网络中的 ReLU (Rectified Linear Unit) 激活层的输出。
# ReLU 激活函数是一种非线性函数，广泛应用于深度学习模型中
def ComputeReluLayer(img):
    """
    对输入数据应用 ReLU 激活函数。

    参数:
    img (numpy.ndarray): 输入数据，可以是神经网络中任意层的输出。

    返回:
    numpy.ndarray: 经过 ReLU 激活函数处理后的结果。
    """
    print("-- compute relu")
    # 应用 ReLU 函数，将所有负值设置为 0
    res = np.maximum(0, img)
    print(res.shape)
    return res


# 这段代码定义了 ComputeBottleNeck 函数，它实现了 ResNet50 网络中的一个关键特性：bottleneck 结构，包括残差连接。
# 这种结构有助于解决深度神经网络中的梯度消失问题，并允许网络学习更深层次的特征。
def ComputeBottleNeck(in_data, bottleneck_layer_name, down_sample=False):
    """
    计算 ResNet50 中的一个 bottleneck 结构，包括残差连接。

    参数:
    in_data (numpy.ndarray): 输入数据。
    bottleneck_layer_name (str): bottleneck 结构的名称。
    down_sample (bool): 指示是否在此结构中应用下采样。默认为 False。

    返回:
    numpy.ndarray: bottleneck 结构的输出。
    """
    print("compute " + bottleneck_layer_name)
    # 第一层卷积
    out = ComputeConvLayer(in_data, bottleneck_layer_name + "_conv1")
    # 第一层批归一化
    out = ComputeBatchNormLayer(out, bottleneck_layer_name + "_bn1")
    # ReLU 激活
    out = ComputeReluLayer(out)

    # 第二层卷积
    out = ComputeConvLayer(out, bottleneck_layer_name + "_conv2")
    # 第二层批归一化
    out = ComputeBatchNormLayer(out, bottleneck_layer_name + "_bn2")
    # ReLU 激活
    out = ComputeReluLayer(out)

    # 第三层卷积
    out = ComputeConvLayer(out, bottleneck_layer_name + "_conv3")
    # 第三层批归一化
    bn_out = ComputeBatchNormLayer(out, bottleneck_layer_name + "_bn3")

    # 如果需要下采样，则在残差路径上应用额外的卷积和批归一化
    if down_sample:
        conv_out = ComputeConvLayer(
            in_data, bottleneck_layer_name + "_downsample_conv2d"
        )
        short_cut_out = ComputeBatchNormLayer(
            conv_out, bottleneck_layer_name + "_downsample_batchnorm"
        )
        bn_out = bn_out + short_cut_out
    else:
        # 否则，直接使用输入数据作为残差路径
        bn_out = bn_out + in_data

    # ReLU 激活
    return ComputeReluLayer(bn_out)


# 这个函数 GetPicList 是图像分类任务中的辅助函数，用于获取指定目录下的图片列表
def GetPicList():
    """
    从指定目录中获取图片文件列表。

    返回:
    list: 包含图片文件路径的列表。
    """
    import os  # 导入操作系统接口库

    pic_dir = "../pics/"  # 设置图片文件夹的路径
    # 获取图片目录下所有文件的路径
    file_to_predict = [pic_dir + f for f in os.listdir(pic_dir)]
    # 为了测试，这里指定了一个特定的图片文件
    file_to_predict = ["../pics/cat.jpg"]
    return file_to_predict


# PreProcess 用于对图像进行预处理
def PreProcess(filename):
    """
    对指定的图像文件进行预处理。

    参数:
    filename (str): 图像文件的路径。

    返回:
    numpy.ndarray: 预处理后的图像数据。
    """
    from PIL import Image  # 导入图像处理库
    from torchvision import transforms  # 导入PyTorch视觉变换库

    # 打开图像文件
    img = Image.open(filename)

    # 定义预处理步骤
    PreProcess = transforms.Compose(
        [
            transforms.Resize(256),  # 首先调整图像大小
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),  # 转换为PyTorch张量
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 标准化
        ]
    )

    # 对图像应用预处理
    input_tensor = PreProcess(img)
    # 增加一个批次维度
    input_batch = input_tensor.unsqueeze(0)

    # 将PyTorch张量转换为numpy数组，并调整维度顺序
    out = np.array(input_batch)
    out = np.transpose(out, (0, 2, 3, 1))
    out = np.reshape(out, (224, 224, 3))
    return out


class Resnet:
    def run(self, img):
        """
        执行 ResNet 模型的前向传播。

        参数:
        img (numpy.ndarray): 预处理后的输入图像。

        返回:
        numpy.ndarray: 模型的输出。
        """
        # 初始卷积层、批归一化和ReLU层
        out = ComputeConvLayer(img, "conv1")
        out = ComputeBatchNormLayer(out, "bn1")
        out = ComputeReluLayer(out)
        out = ComputeMaxPoolLayer(out)

        # 通过四个残差层（每层有多个残差块）处理数据
        # layer1
        out = ComputeBottleNeck(out, "layer1_bottleneck0", down_sample=True)
        out = ComputeBottleNeck(out, "layer1_bottleneck1", down_sample=False)
        out = ComputeBottleNeck(out, "layer1_bottleneck2", down_sample=False)

        # layer2
        out = ComputeBottleNeck(out, "layer2_bottleneck0", down_sample=True)
        out = ComputeBottleNeck(out, "layer2_bottleneck1", down_sample=False)
        out = ComputeBottleNeck(out, "layer2_bottleneck2", down_sample=False)
        out = ComputeBottleNeck(out, "layer2_bottleneck3", down_sample=False)

        # layer3
        out = ComputeBottleNeck(out, "layer3_bottleneck0", down_sample=True)
        out = ComputeBottleNeck(out, "layer3_bottleneck1", down_sample=False)
        out = ComputeBottleNeck(out, "layer3_bottleneck2", down_sample=False)
        out = ComputeBottleNeck(out, "layer3_bottleneck3", down_sample=False)
        out = ComputeBottleNeck(out, "layer3_bottleneck4", down_sample=False)
        out = ComputeBottleNeck(out, "layer3_bottleneck5", down_sample=False)

        # layer4
        out = ComputeBottleNeck(out, "layer4_bottleneck0", down_sample=True)
        out = ComputeBottleNeck(out, "layer4_bottleneck1", down_sample=False)
        out = ComputeBottleNeck(out, "layer4_bottleneck2", down_sample=False)

        # 平均池化和全连接层
        out = ComputeAvgPoolLayer(out)
        out = ComputeFcLayer(out, "fc")
        return out


# 导入时间模块,用来计算模型推理时间
import time

if __name__ == "__main__":
    # 获取待预测的图像列表
    pics = GetPicList()

    # 创建 Resnet 实例
    module = Resnet()

    total_time = 0.0  # 初始化总时间
    for filename in pics:
        print("Begin predict with " + filename)
        # 预处理图像
        pre_out = PreProcess(filename)

        # 开始计时
        start = time.time()
        # 执行模型前向传播
        res = module.run(pre_out)
        # 结束计时
        end = time.time()
        # 累计时间
        total_time = total_time + end - start

        # 获取并打印预测结果
        out_res = list(res)
        max_value = max(out_res)
        index = out_res.index(max_value)

        print("\npredict picture: " + filename)
        print("      max_value: " + str(max_value))
        print("          index: " + str(index))

        # 读取类别名称并打印预测类别
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
            print("         result: " + categories[index])

    # 计算平均延迟和吞吐量
    total_time = total_time * 60  # 将时间转换为毫秒
    latency = total_time / len(pics)
    print("\033[0;32mAverage Latency : ", latency, "ms \033[0m")
    print("\033[0;32mAverage Throughput : ", (1000 / latency), "fps \033[0m")
