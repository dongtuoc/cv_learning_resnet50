import numpy as np


def Visualization(img):
    """
    可视化图像的不同通道。

    参数:
    img (numpy.ndarray): 输入的多通道图像。
    """
    import matplotlib.pyplot as plt  # 导入 Matplotlib 库

    # 创建子图布局
    fig, ax = plt.subplots(1, 4, figsize=(12, 16))
    # 可视化图像的特定通道
    ax[0].matshow(img[:, :, 0], cmap="viridis")  # 显示第一个通道
    ax[1].matshow(img[:, :, 5], cmap="viridis")  # 显示第六个通道
    ax[2].matshow(img[:, :, 11], cmap="viridis")  # 显示第十二个通道
    ax[3].matshow(img[:, :, 24], cmap="viridis")  # 显示第二十五个通道

    plt.savefig("a.png")  # 将可视化结果保存为图片


def Conv2d(img, weight, hi, wi, ci, co, kernel, stride, pad):
    """
    执行2D卷积操作。

    参数:
    img (numpy.ndarray): 输入图像。
    weight (list): 卷积核的权重。
    hi, wi (int): 输入图像的高度和宽度。
    ci, co (int): 输入和输出通道数。
    kernel (int): 卷积核大小。
    stride (int): 卷积步长。
    pad (int): 边缘填充大小。

    返回:
    numpy.ndarray: 卷积后的输出图像。
    """
    # 计算输出图像的尺寸
    ho = (hi + 2 * pad - kernel) // stride + 1
    wo = (wi + 2 * pad - kernel) // stride + 1

    # 转换权重格式并对图像进行填充
    weight = np.array(weight).reshape(co, kernel, kernel, ci)
    img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), "constant")
    img_out = np.zeros((ho, wo, co))

    # 执行卷积操作
    for co_ in range(co):
        for ho_ in range(ho):
            in_h_origin = ho_ * stride - pad
            for wo_ in range(wo):
                in_w_origin = wo_ * stride - pad
                # 对每个卷积窗口进行计算
                filter_h_start = max(0, -in_h_origin)
                filter_w_start = max(0, -in_w_origin)
                filter_h_end = min(kernel, hi - in_h_origin)
                filter_w_end = min(kernel, wi - in_w_origin)
                acc = float(0)
                for kh_ in range(filter_h_start, filter_h_end):
                    hi_index = in_h_origin + kh_
                    for kw_ in range(filter_w_start, filter_w_end):
                        wi_index = in_w_origin + kw_
                        for ci_ in range(ci):
                            in_data = img[hi_index][wi_index][ci_]
                            weight_data = weight[co_][kh_][kw_][ci_]
                            acc = acc + in_data * weight_data
                img_out[ho_][wo_][co_] = acc
    return img_out  # 返回卷积后的图像


def Conv2dOpt(img, weight, hi, wi, ci, co, kernel, stride, pad):
    """
    执行2D卷积操作，使用优化技巧提高性能。

    参数:
    同 Conv2d 函数。

    返回:
    numpy.ndarray: 卷积后的输出图像。
    """
    # 计算输出图像的尺寸
    ho = (hi + 2 * pad - kernel) // stride + 1
    wo = (wi + 2 * pad - kernel) // stride + 1

    # 转换权重格式并对图像进行填充
    weight = np.array(weight).reshape(co, kernel, kernel, ci)
    img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), "constant")
    img_out = np.zeros((ho, wo, co))


    # 使用 vdot 来优化乘加操作（MAC）
    for co_ in range(co):
        for ho_ in range(ho):
            in_h_origin = ho_ * stride - pad
            for wo_ in range(wo):
                in_w_origin = wo_ * stride - pad
                filter_h_start = max(0, -in_h_origin)
                filter_w_start = max(0, -in_w_origin)
                filter_h_end = min(kernel, hi - in_h_origin)
                filter_w_end = min(kernel, wi - in_w_origin)
                acc = float(0)
                for kh_ in range(filter_h_start, filter_h_end):
                    hi_index = in_h_origin + kh_
                    for kw_ in range(filter_w_start, filter_w_end):
                        wi_index = in_w_origin + kw_
                        # use vdot to optimize MAC operation
                        acc += np.vdot(img[hi_index][wi_index], weight[co_][kh_][kw_])
                img_out[ho_][wo_][co_] = acc

    return img_out
