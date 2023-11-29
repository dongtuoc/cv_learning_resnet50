import numpy as np
from torchvision import models
import torch

resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet50.eval()
print(resnet50)

# download onnx file of resnet50
# torch.onnx.export(resnet50, torch.randn(1, 3, 224, 224), "./resnet50.onnx")

dump_dir = "./resnet50_weight/"

def save_conv_param(data, file):
  kh = data.kernel_size[0]
  sh = data.stride[0]
  pad_l = data.padding[0]
  ci = data.in_channels
  co = data.out_channels
  l = [ci, co, kh, sh, pad_l]
  np.savetxt(dump_dir + file + str("_param.txt"), l)

def save_bn_param(data, file):
  eps = data.eps
  momentum = data.momentum
  l = [eps, momentum]
  np.savetxt(dump_dir + file + str("_param.txt"), l)

def save(data, file):
  if isinstance(data, type(resnet50.conv1)):
    save_conv_param(data, file)
    # weight in model is [co, ci, kh, kw]
    # my compute use [co, kh, kw, ci]
    w = np.array(data.weight.data.cpu().numpy())
    w = np.transpose(w, (0, 2, 3, 1))
    np.savetxt(dump_dir + file + str("_weight.txt"), w.reshape(-1, 1))

  if isinstance(data, type(resnet50.bn1)):
    save_bn_param(data, file)
    m = np.array(data.running_mean.data.cpu().numpy())
    np.savetxt(dump_dir + file + str("_running_mean.txt"), m.reshape(-1, 1))

    v = np.array(data.running_var.data.cpu().numpy())
    np.savetxt(dump_dir + file + str("_running_var.txt"), v.reshape(-1, 1))

    b = np.array(data.bias.data.cpu().numpy())
    np.savetxt(dump_dir + file + str("_bias.txt"), b.reshape(-1, 1))

    w = np.array(data.weight.data.cpu().numpy())
    np.savetxt(dump_dir + file + str("_weight.txt"), w.reshape(-1, 1))

  if isinstance(data, type(resnet50.fc)):
    print(data.weight.shape)
    bias = np.array(data.bias.data.cpu().numpy())
    np.savetxt(dump_dir + file + str("_bias.txt"), bias.reshape(-1, 1))

    w = np.array(data.weight.data.cpu().numpy())
    np.savetxt(dump_dir + file + str("_weight.txt"), w.reshape(-1, 1))

def save_bottle_neck(layer, layer_index):
  bottle_neck_idx = 0

  layer_name = "resnet50_layer" + str(layer_index) + "_bottleneck"
  for bottleNeck in layer:
    save(bottleNeck.conv1, layer_name + str(bottle_neck_idx) + "_conv1")
    save(bottleNeck.bn1, layer_name + str(bottle_neck_idx) + "_bn1")
    save(bottleNeck.conv2, layer_name + str(bottle_neck_idx) + "_conv2")
    save(bottleNeck.bn2, layer_name + str(bottle_neck_idx) + "_bn2")
    save(bottleNeck.conv3, layer_name + str(bottle_neck_idx) + "_conv3")
    save(bottleNeck.bn3, layer_name + str(bottle_neck_idx) + "_bn3")
    if bottleNeck.downsample:
      save(bottleNeck.downsample[0], layer_name + str(bottle_neck_idx) + "_downsample_conv2d")
      save(bottleNeck.downsample[1], layer_name + str(bottle_neck_idx) + "_downsample_batchnorm")
    bottle_neck_idx = bottle_neck_idx + 1


save(resnet50.conv1, "resnet50_conv1")
save(resnet50.bn1, "resnet50_bn1")

save_bottle_neck(resnet50.layer1, 1)
save_bottle_neck(resnet50.layer2, 2)
save_bottle_neck(resnet50.layer3, 3)
save_bottle_neck(resnet50.layer4, 4)

save(resnet50.fc, "resnet50_fc")

