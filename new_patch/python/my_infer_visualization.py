import matplotlib.pyplot as plt
import numpy as np

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
            # use vdot to optimize MAC operation
            acc += np.vdot(img[hi_index][wi_index], weight[co_][kh_][kw_])
            #for ci_ in range(ci):
            #  in_data = img[hi_index][wi_index][ci_]
            #  weight_data = weight[co_][kh_][kw_][ci_]
            #  acc = acc + in_data * weight_data
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
    # use vdot to optimize MAC operation
    sum_x = np.vdot(img_new, weight_new[i])
    out[i] = sum_x + bias_new[i]
    #sum_x = float(0)
    #for j in range(2048):
    #  l = img_new[j]
    #  r = weight_new[i][j]
    #  sum_x = sum_x + l * r
    #out[i] = sum_x + bias_new[i]
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

def my_bn(img, mean, var, gamma, bias):
  h = img.shape[0]
  w = img.shape[1]
  c = img.shape[2]

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
    k = [float(l) for l in lines ]
    if is_float == False:
      k = [int(l) for l in k]
    #if is_float == True:
    #  for l in lines:
    #    k.append(float(l))
    #else:
    #  for l in lines:
    #    k.append(int(float(l)))
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

# get pics from `../../pics`
def getPicList():
  import os
  pic_dir = "../pics/"
  file_to_predict = [pic_dir + f for f in os.listdir(pic_dir)]
  file_to_predict = ["../pics/cat.jpg"]
  return file_to_predict

# pre-process for pictures
def preprocess(filename):
  from PIL import Image
  from torchvision import transforms
  img = Image.open(filename)
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(img)
  input_batch = input_tensor.unsqueeze(0)
  
  out = np.array(input_batch)
  out = np.transpose(out, (0, 2, 3, 1))
  out = np.reshape(out, (224, 224, 3))
  return out

def visualizaiton(feature_map):
  channel=feature_map.shape[2]
  row = 8
  col = int(channel / 8)
  fig, axes = plt.subplots(nrows=row, ncols=col)
  for i in range(row):
    for j in range(col):
      axes[i][j].matshow(feature_map[:,:,int(i*col+j)])
  plt.show()
  time.sleep(100)
  exit()
  

import time
pic_to_predice = getPicList()
# Resnet50 constructor is showed layer by layer as follows.
for filename in pic_to_predice:
  print("begin predice with " + filename)
  out = preprocess(filename)

  out = compute_conv_layer(out, "conv1")
  # visualizaiton(out)
  out = compute_bn_layer(out, "bn1")
  # print("-----------------------bn -------------")
  # print(out)
  # exit()

  out = compute_relu_layer(out)
  out = compute_maxpool_layer(out)

  # layer1 
  out = compute_bottleneck(out, "layer1_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer1_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer1_bottleneck2", down_sample = False)

  # layer2
  out = compute_bottleneck(out, "layer2_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer2_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer2_bottleneck2", down_sample = False)
  out = compute_bottleneck(out, "layer2_bottleneck3", down_sample = False)

  # layer3
  out = compute_bottleneck(out, "layer3_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer3_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck2", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck3", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck4", down_sample = False)
  out = compute_bottleneck(out, "layer3_bottleneck5", down_sample = False)
  
  # layer4
  out = compute_bottleneck(out, "layer4_bottleneck0", down_sample = True)
  out = compute_bottleneck(out, "layer4_bottleneck1", down_sample = False)
  out = compute_bottleneck(out, "layer4_bottleneck2", down_sample = False)

  # avg pool
  out = compute_avgpool_layer(out)
  # Linear
  out = compute_fc_layer(out, "fc")

  # find inference result
  out_res = list(out)
  max_value = max(out_res)
  index = out_res.index(max_value)
  
  print("\npredict picture: " + filename)
  print("      max_value: " + str(max_value))
  print("          index: " + str(index))
  
  # Read the categories
  with open("imagenet_classes.txt", "r") as f:
      categories = [s.strip() for s in f.readlines()]
      print("         result: " + categories[index])
