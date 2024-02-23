import numpy as np

params_dir = "../model/resnet50_weight/resnet50_"

# if The Programe costs too long time to infer your picture, try set `use_opt` to True
# to enable `np.dot`, which can optimize convolution MAC(Mul-Add Compute)
# calculating.
use_opt = True


# load data from txt
def LoadDataFromFile(file_name, is_float=True):
    k = []
    with open(file_name, "r") as f_:
        lines = f_.readlines()
        k = [float(l) for l in lines]
        if is_float == False:
            k = [int(l) for l in k]
    return k


def LoadConvWeight(name):
    name = params_dir + name + "_weight.txt"
    return LoadDataFromFile(name, is_float=True)


def LoadConvParam(name):
    name = params_dir + name + "_param.txt"
    param = LoadDataFromFile(name, is_float=False)
    return param


import ops.conv2d as conv


def ComputeConvLayer(in_data, layer_name):
    print("-- compute " + layer_name)
    weight = LoadConvWeight(layer_name)
    param = LoadConvParam(layer_name)
    # ci, co, kernel, stride, pad
    hi = in_data.shape[0]
    wi = in_data.shape[1]
    ci = param[0]
    co = param[1]
    kernel = param[2]
    stride = param[3]
    pad = param[4]
    if use_opt:
        res = conv.Conv2dOpt(in_data, weight, hi, wi, ci, co, kernel, stride, pad)
    else:
        res = conv.Conv2d(in_data, weight, hi, wi, ci, co, kernel, stride, pad)
    print(res.shape)
    return res


import ops.fc as fc


def ComputeFcLayer(in_data, layer_name):
    print("-- compute " + layer_name)
    weight_file_name = params_dir + layer_name + "_weight.txt"
    bias_file_name = params_dir + layer_name + "_bias.txt"
    weight = LoadDataFromFile(weight_file_name)
    bias = LoadDataFromFile(bias_file_name)
    if use_opt:
        res = fc.FullyConnectOpt(in_data, weight, bias)
    else:
        res = fc.FullyConnect(in_data, weight, bias)
    print(res.shape)
    return res


import ops.bn as bn


def ComputeBatchNormLayer(in_data, layer_name):
    print("-- compute " + layer_name)
    weight = LoadConvWeight(layer_name)
    weight_file_name = params_dir + layer_name + "_weight.txt"
    bias_file_name = params_dir + layer_name + "_bias.txt"
    mean_file_name = params_dir + layer_name + "_running_mean.txt"
    var_file_name = params_dir + layer_name + "_running_var.txt"
    weight = LoadDataFromFile(weight_file_name)
    bias = LoadDataFromFile(bias_file_name)
    mean = LoadDataFromFile(mean_file_name)
    var = LoadDataFromFile(var_file_name)
    res = bn.BatchNorm(in_data, mean, var, weight, bias)
    print(res.shape)
    return res


import ops.pool as pool


def ComputeMaxPoolLayer(in_data):
    print("-- compute maxpool")
    res = pool.MaxPool(in_data)
    print(res.shape)
    return res


def ComputeAvgPoolLayer(in_data):
    print("-- compute avgpool")
    res = pool.AvgPool(in_data)
    print(res.shape)
    return res


def ComputeReluLayer(img):
    print("-- compute relu")
    res = np.maximum(0, img)
    print(res.shape)
    return res


def ComputeBottleNeck(in_data, bottleneck_layer_name, down_sample=False):
    print("compute " + bottleneck_layer_name)
    out = ComputeConvLayer(in_data, bottleneck_layer_name + "_conv1")
    out = ComputeBatchNormLayer(out, bottleneck_layer_name + "_bn1")
    out = ComputeReluLayer(out)
    out = ComputeConvLayer(out, bottleneck_layer_name + "_conv2")
    out = ComputeBatchNormLayer(out, bottleneck_layer_name + "_bn2")
    out = ComputeReluLayer(out)
    out = ComputeConvLayer(out, bottleneck_layer_name + "_conv3")
    bn_out = ComputeBatchNormLayer(out, bottleneck_layer_name + "_bn3")

    if down_sample == True:
        conv_out = ComputeConvLayer(
            in_data, bottleneck_layer_name + "_downsample_conv2d"
        )
        short_cut_out = ComputeBatchNormLayer(
            conv_out, bottleneck_layer_name + "_downsample_batchnorm"
        )
        bn_out = bn_out + short_cut_out
    else:
        bn_out = bn_out + in_data
    return ComputeReluLayer(bn_out)


# get pics from `../../pics`
def GetPicList():
    import os

    pic_dir = "../pics/"
    file_to_predict = [pic_dir + f for f in os.listdir(pic_dir)]
    file_to_predict = ["../pics/cat.jpg"]
    return file_to_predict


# pre-process for pictures
def PreProcess(filename):
    from PIL import Image
    from torchvision import transforms

    img = Image.open(filename)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 3, figsize=(12,16))

    PreProcess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = PreProcess(img)
    input_batch = input_tensor.unsqueeze(0)

    out = np.array(input_batch)
    out = np.transpose(out, (0, 2, 3, 1))
    out = np.reshape(out, (224, 224, 3))
    # ax[0].matshow(out[:, :, 0], cmap="viridis")
    # ax[1].matshow(out[:, :, 1], cmap="viridis")
    # ax[2].matshow(out[:, :, 2], cmap="viridis")
    # plt.savefig("a.png")
    return out


class Resnet:
    def run(self, img):
        out = ComputeConvLayer(img, "conv1")
        out = ComputeBatchNormLayer(out, "bn1")
        out = ComputeReluLayer(out)
        out = ComputeMaxPoolLayer(out)

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

        # avg pool
        out = ComputeAvgPoolLayer(out)
        # Linear
        out = ComputeFcLayer(out, "fc")
        return out


import time

if __name__ == "__main__":
    pics = GetPicList()

    module = Resnet()

    total_time = 0.0
    for filename in pics:
        print("Begin predict with " + filename)
        pre_out = PreProcess(filename)

        start = time.time()
        res = module.run(pre_out)
        end = time.time()
        total_time = total_time + end - start

        # find inference result
        out_res = list(res)
        max_value = max(out_res)
        index = out_res.index(max_value)

        print("\npredict picture: " + filename)
        print("      max_value: " + str(max_value))
        print("          index: " + str(index))

        # Read the categories
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
            print("         result: " + categories[index])

    total_time = total_time * 60  # convert to ms
    latency = total_time / len(pics)
    print("\033[0;32mAverage Latency : ", latency, "ms \033[0m")
    print("\033[0;32mAverage Throughput : ", (1000 / latency), "fps \033[0m")
