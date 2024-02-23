import numpy as np


def FullyConnect(img, weight, bias):
    """
    执行全连接层的计算。

    参数:
    img (numpy.ndarray): 来自上一层的输入。
    weight (list): 全连接层的权重。
    bias (list): 全连接层的偏置。

    返回:
    numpy.ndarray: 全连接层的输出。
    """
    # reshape输入为一维数组
    img_new = img.reshape(2048)
    # reshape权重为 [1000, 2048]
    weight_new = np.array(weight).reshape([1000, 2048])
    # reshape偏置为 [1000]
    bias_new = np.array(bias).reshape(1000)
    # 初始化输出数组
    out = np.zeros(1000)

    # 对每个输出单元执行加权求和和偏置添加
    for i in range(1000):
        sum_x = float(0)
        for j in range(2048):
            l = img_new[j]
            r = weight_new[i][j]
            sum_x = sum_x + l * r
        out[i] = sum_x + bias_new[i]
    return out  # 返回全连接层的输出


def FullyConnectOpt(img, weight, bias):
    """
    使用优化版本执行全连接层的计算。

    参数:
    同 FullyConnect 函数。

    返回:
    numpy.ndarray: 全连接层的输出。
    """
    # reshape输入为一维数组
    img_new = img.reshape(2048)
    # reshape权重为 [1000, 2048]
    weight_new = np.array(weight).reshape([1000, 2048])
    # reshape偏置为 [1000]
    bias_new = np.array(bias).reshape(1000)
    # 初始化输出数组
    out = np.zeros(1000)

    # 使用 np.vdot 来优化乘加操作（MAC）
    for i in range(1000):
        sum_x = np.vdot(img_new, weight_new[i])
        out[i] = sum_x + bias_new[i]
    return out
