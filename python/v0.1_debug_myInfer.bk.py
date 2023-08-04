import numpy as np
from PIL import Image
import datetime
from torchvision import transforms

# we just use NHWC to calculate
# no dilation
def my_conv2d(img, weight, hi, wi, ci, co, kernel, stride, pad):
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1
  
  weight = np.array(weight).reshape(co, kernel, kernel, ci)
  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, co))

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
            for ci_ in range(ci):
              in_data = img[hi_index][wi_index][ci_]
              weight_data = weight[co_][kh_][kw_][ci_]
              acc = acc + in_data * weight_data
        img_out[ho_][wo_][co_] = acc
  return img_out

def my_fc(img, weight, bias):
  '''
  fc compute [2048] * [1000, 2048] = [1000]
  img : [1, 1, 2048] from last layer
  weight: need reshpe to [1000, 2048]
  bias: [1000]
  '''
  img_new = img.reshape(2048)
  weight_new = np.array(weight).reshape([1000, 2048])
  bias_new = np.array(bias).reshape(1000)
  out = np.zeros(1000)
  for i in range(1000):
    sum_x = float(0)
    for j in range(2048):
      l = img_new[j]
      r = weight_new[i][j]
      sum_x = sum_x + l * r
    out[i] = sum_x + bias_new[i]
  return out

def my_max_pool(img):
  hi  = img.shape[0]
  wi = img.shape[1]
  channel = img.shape[2]
  pad = 1
  stride = 2
  kernel = 3
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1
  
  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, channel))

  for c_ in range(channel):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        max_x = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            in_data = img[hi_index][wi_index][c_]
            max_x = max(in_data, max_x)
        img_out[ho_][wo_][c_] = max_x 
  return img_out

def my_avg_pool(img):
  hi  = img.shape[0]
  wi = img.shape[1]
  channel = img.shape[2]
  pad = 0
  stride = 1
  kernel = 7
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1

  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, channel))

  for c_ in range(channel):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        sum_x = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            in_data = img[hi_index][wi_index][c_]
            sum_x = sum_x + in_data
        img_out[ho_][wo_][c_] = sum_x / (kernel * kernel)
  return img_out

def my_mean(x):
  sum_x = float(0)
  x = x.reshape(x.size)
  for i in x:
    sum_x = sum_x + i
  return sum_x / x.size

def my_var(x):
  mean = my_mean(x)
  var_x = float(0)
  x = x.reshape(x.size)
  for i in x:
    var_x = var_x + pow((i - mean), 2)
  return var_x / x.size

def my_bn(img, mean, var, gamma, bias):
  h = img.shape[0]
  w = img.shape[1]
  c = img.shape[2]
  
  #print(mean)
  ## HWC
  ##mu = np.mean(img, axis = (0, 1))
  ##var = np.var(img, axis = (0, 1))

  #img_norm = (img - mean) / np.sqrt(var + 1e-5)
  #bn = gamma * img_norm + bias 
  #return bn

  for c_ in range(c):
    data = img[:, :, c_]
    data_ = (data - mean[c_]) / (pow(var[c_] + 1e-5, 0.5))
    data_ = data_ * gamma[c_]
    data_ = data_ + bias[c_]
    img[:, :, c_] = data_
  return img
    
def compute_relu_layer(img):
  print("-- compute relu")
  res = np.maximum(0, img)
  print(res.shape)
  return res

# load data from txt
def load_data_from_file(file_name, is_float = True):
  k = []
  with open(file_name, 'r') as f_:
    lines = f_.readlines()
    if is_float == True:
      for l in lines:
        k.append(float(l))
    else:
      for l in lines:
        k.append(int(float(l)))
  return k

def load_conv_weight(name):
  name = "../model/resnet50_weight/resnet50_" + name + "_weight.txt"
  return load_data_from_file(name, is_float = True)

def load_conv_param(name):
  name = "../model/resnet50_weight/resnet50_" + name + "_param.txt"
  param = load_data_from_file(name, is_float = False)
  return param

def compute_conv_layer(in_data, layer_name):
  print("-- compute " + layer_name)
  weight = load_conv_weight(layer_name)
  param = load_conv_param(layer_name)
  # ci, co, kernel, stride, pad
  hi = in_data.shape[0]
  wi = in_data.shape[1]
  ci = param[0]
  co = param[1]
  kernel = param[2]
  stride = param[3]
  pad = param[4]
  res = my_conv2d(in_data, weight, hi, wi, ci, co, kernel, stride, pad)
  print(res.shape)
  return res

def compute_fc_layer(in_data, layer_name):
  print("-- compute " + layer_name)
  weight_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt"
  bias_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt"
  weight = load_data_from_file(weight_file_name)
  bias = load_data_from_file(bias_file_name)
  res = my_fc(in_data, weight, bias)
  print(res.shape)
  return res

def compute_bn_layer(in_data, layer_name):
  print("-- compute " + layer_name)
  weight = load_conv_weight(layer_name)
  weight_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt"
  bias_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt"
  mean_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt"
  var_file_name = "../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt"
  weight = load_data_from_file(weight_file_name)
  bias = load_data_from_file(bias_file_name)
  mean = load_data_from_file(mean_file_name)
  var = load_data_from_file(var_file_name)
  res = my_bn(in_data, mean, var, weight, bias)  
  print(res.shape)
  return res

def compute_maxpool_layer(in_data):
  print("-- compute maxpool")
  res = my_max_pool(in_data)  
  print(res.shape)
  return res

def compute_avgpool_layer(in_data):
  print("-- compute avgpool")
  res = my_avg_pool(in_data)
  print(res.shape)
  return res

def compute_bottleneck(in_data, bottleneck_layer_name, down_sample = False):
  print("compute " + bottleneck_layer_name)
  out = compute_conv_layer(in_data, bottleneck_layer_name + "_conv1")
  out = compute_bn_layer(out, bottleneck_layer_name + "_bn1")
  out = compute_relu_layer(out)
  out = compute_conv_layer(out, bottleneck_layer_name + "_conv2")
  out = compute_bn_layer(out, bottleneck_layer_name + "_bn2")
  out = compute_relu_layer(out)
  out = compute_conv_layer(out, bottleneck_layer_name + "_conv3")
  bn_out = compute_bn_layer(out, bottleneck_layer_name + "_bn3")

  if down_sample == True:
    conv_out= compute_conv_layer(in_data, bottleneck_layer_name + "_downsample_conv2d")
    short_cut_out = compute_bn_layer(conv_out, bottleneck_layer_name + "_downsample_batchnorm")
    bn_out = bn_out + short_cut_out
  else:
    bn_out = bn_out + in_data
  return compute_relu_layer(bn_out)

def debug_func(golden, res):
  '''
  to compare golden and my_res
  note: golden is layout NCHW, my_res is layout NHWC, so golden should be transposed.
  '''
  golden_trans = np.transpose(golden.detach().numpy(), (0, 2, 3, 1))
  golden_list = list(golden_trans.reshape(-1, 1))
  res_list = list(res.reshape(-1, 1))
  for i in range(len(res_list)):
    if abs(golden_list[i] - res_list[i]) > 0.01:
       print("Error at " + str(i))
       print("golden = " + str(golden_list[i]) + ", res = " + str(res_list[i]))
       return
  print("succ")
  
def debug_func_for_fc(golden, res):
  golden_list = list(golden.detach().numpy().reshape(-1, 1))
  res_list = list(res.reshape(-1, 1))
  for i in range(len(res_list)):
    if abs(golden_list[i] - res_list[i]) > 0.01:
       print("Error at " + str(i))
       print("golden = " + str(golden_list[i]) + ", res = " + str(res_list[i]))
       return
  print("succ")



# read picture and do pre-process
filename = "../pics/cat.jpg"
img = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
#print(input_batch)

out = np.array(input_batch)
out = np.transpose(out, (0, 2, 3, 1))
out = np.reshape(out, (224, 224, 3))
#print(out.shape)
#print(type(out))


import torch
resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet50.eval()
print(resnet50)

out = compute_conv_layer(out, "conv1")

print("conv1")
net00 = torch.nn.Sequential(resnet50.conv1)
debug_func(net00(input_batch), out)

out = compute_bn_layer(out, "bn1")
print("bn1")
bn_net= torch.nn.Sequential(resnet50.conv1, resnet50.bn1)
debug_func(bn_net(input_batch), out)

out = compute_relu_layer(out)
out = compute_maxpool_layer(out)

print("maxpool")
net0 = torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool)
debug_func(net0(input_batch), out)

# layer1 
out = compute_bottleneck(out, "layer1_bottleneck0", down_sample = True)
out = compute_bottleneck(out, "layer1_bottleneck1", down_sample = False)
out = compute_bottleneck(out, "layer1_bottleneck2", down_sample = False)

print("layer1")
net1 = torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1)
debug_func(net1(input_batch), out)

# layer2
out = compute_bottleneck(out, "layer2_bottleneck0", down_sample = True)
out = compute_bottleneck(out, "layer2_bottleneck1", down_sample = False)
out = compute_bottleneck(out, "layer2_bottleneck2", down_sample = False)
out = compute_bottleneck(out, "layer2_bottleneck3", down_sample = False)

print("layer2")
net2 = torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2)
debug_func(net2(input_batch), out)

# layer3
out = compute_bottleneck(out, "layer3_bottleneck0", down_sample = True)
out = compute_bottleneck(out, "layer3_bottleneck1", down_sample = False)
out = compute_bottleneck(out, "layer3_bottleneck2", down_sample = False)
out = compute_bottleneck(out, "layer3_bottleneck3", down_sample = False)
out = compute_bottleneck(out, "layer3_bottleneck4", down_sample = False)
out = compute_bottleneck(out, "layer3_bottleneck5", down_sample = False)

print("layer3")
net3= torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.layer3)
debug_func(net3(input_batch), out)

# layer4
out = compute_bottleneck(out, "layer4_bottleneck0", down_sample = True)
out = compute_bottleneck(out, "layer4_bottleneck1", down_sample = False)
out = compute_bottleneck(out, "layer4_bottleneck2", down_sample = False)

print("layer4")
net4= torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4)
debug_func(net4(input_batch), out)

# avg pool
out = compute_avgpool_layer(out)

net = torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool,
                          resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4,
                          resnet50.avgpool)
net_out = net(input_batch)
print("avgpool")
debug_func(net_out, out)

# Linear
out = compute_fc_layer(out, "fc")

net1 = torch.nn.Sequential(resnet50.fc)
print("fc")
out_seq = net1(net_out.flatten())
debug_func_for_fc(out_seq, out)
print(out_seq)
print("=======================================")

#---------------------- find inference result -----------------#
out_res = list(out)
print(out_res)
max_value = max(out_res)
index = out_res.index(max_value)

print("\npredict picture: " + filename)
print("      max_value: " + str(max_value))
print("          index: " + str(index))

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    print("         result: " + categories[index])
