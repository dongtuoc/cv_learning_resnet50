def BatchNorm(img, mean, var, gamma, bias):
    """
    对图像进行批量归一化处理。

    参数:
    img (numpy.ndarray): 输入图像。
    mean (numpy.ndarray): 归一化处理所用的均值。
    var (numpy.ndarray): 归一化处理所用的方差。
    gamma (numpy.ndarray): 缩放参数。
    bias (numpy.ndarray): 偏移参数。

    返回:
    numpy.ndarray: 批量归一化后的图像。
    """
    # 获取图像的高度、宽度和通道数
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]

    # 对每个通道进行批量归一化处理
    for c_ in range(c):
        # 提取单个通道的数据
        data = img[:, :, c_]
        # 执行批量归一化：(data - mean) / sqrt(var + epsilon)
        data_ = (data - mean[c_]) / (pow(var[c_] + 1e-5, 0.5))
        # 应用缩放和偏移
        data_ = data_ * gamma[c_]
        data_ = data_ + bias[c_]
        # 更新图像的该通道数据
        img[:, :, c_] = data_

    return img  # 返回批量归一化后的图像
