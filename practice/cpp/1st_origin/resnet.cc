#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "ops/bn.h"
#include "ops/conv2d.h"
#include "ops/fc.h"
#include "ops/pool.h"

// 模板函数，用于从文件中加载数据
template <typename T>
static T* load_data_from_file(const std::string& file_name, int len, bool is_float) {
  // 动态分配内存以存储数据
  T* data = (T*)malloc(len * sizeof(T));
  // 打开文件
  FILE* fp = fopen(file_name.c_str(), "r");

  // 逐个读取数据
  for (auto i = 0; i < len; i++) {
    float x = 0;
    fscanf(fp, "%f", &x);             // 读取浮点数
    data[i] = is_float ? x : (int)x;  // 根据数据的类型进行转换并且存储
  }

  // 关闭文件
  fclose(fp);
  return data;  // 返回从文件中加载的数据
}

// 加载卷积层的权重
static float* load_conv_weight(const std::string& name, int len) {
  // 构建权重文件的完整路径
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  // 调用 load_data_from_file 函数读取权重
  return load_data_from_file<float>(file_name, len, true);
}

// 加载卷积层的参数
static int* load_conv_param(const std::string& name, int len) {
  // 构建参数文件的完整路径
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  // 调用 load_data_from_file 函数读取参数
  return load_data_from_file<int>(file_name, len, false);
}

// 实现 ReLU 激活函数
float* compute_relu_layer(float* img, int len) {
  // 遍历输入数据
  for (int i = 0; i < len; i++) {
    // 应用 ReLU，即如果元素值小于0，则置为0
    img[i] = img[i] > 0 ? img[i] : 0;
  }
  return img;  // 返回激活后的数据
}

// 计算卷积层
float* compute_conv_layer(float* img,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& layer_name,
                          bool is_free_img = true) {
  // 加载卷积层参数：输入通道数、输出通道数、卷积核大小、步长和填充
  auto param = load_conv_param(layer_name, 5);
  auto ci = param[0];      // 输入通道数
  co = param[1];           // 输出通道数
  auto kernel = param[2];  // 卷积核大小
  auto stride = param[3];  // 步长
  auto pad = param[4];     // 填充

  // 加载卷积权重
  auto weight = load_conv_weight(layer_name, co * kernel * kernel * ci);

  // 调用 ops/conv.cc 中的卷积函数，执行卷积运算
  return my_conv2d(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, is_free_img);
}

// 计算全连接层
float* compute_fc_layer(float* img, const std::string& layer_name) {
  // 构建权重和偏置文件的完整路径
  std::string weight_file_name =
      "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  std::string bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";

  // 加载权重和偏置
  auto weight = load_data_from_file<float>(weight_file_name, 1000 * 2048, true);
  auto bias = load_data_from_file<float>(bias_file_name, 1000, true);

  // 执行全连接运算
  return my_fc(img, weight, bias);
}

// 计算批量归一化层
float* compute_bn_layer(float* in_data, int h, int w, int c, const std::string& layer_name) {
  // 构建批量归一化层权重、偏置、均值和方差文件的完整路径
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";

  // 加载批量归一化层的参数
  auto gamma = load_data_from_file<float>(weight_file_name, c, true);
  auto bias = load_data_from_file<float>(bias_file_name, c, true);
  auto mean = load_data_from_file<float>(mean_file_name, c, true);
  auto var = load_data_from_file<float>(var_file_name, c, true);

  // 执行批量归一化运算
  return my_bn(in_data, mean, var, gamma, bias, h, w, c);
}

// 计算最大池化层
float* compute_maxpool_layer(float* in_data) {
  // 调用 my_max_pool 函数来执行最大池化操作
  return my_max_pool(in_data);
}

// 计算平均池化层
float* compute_avgpool_layer(float* in_data) {
  // 调用 my_avg_pool 函数来执行平均池化操作
  return my_avg_pool(in_data);
}

// 计算bottleneck层，bottleneck层通常由三到四个不同的conv + BN + RELU 结构组成，并且 bottleneck
// 结构可能包含下采样 bottleneck 结构参考本仓库 model/resnet50.onnx.png 文件
float* compute_bottleneck(float* in_data,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& bottleneck_layer_name,
                          bool down_sample) {
  int h0, w0, c0;
  int h1, w1, c1;

  // 第一个 conv + BN + RELU 结构
  auto out =
      compute_conv_layer(in_data, hi, wi, h0, w0, c0, bottleneck_layer_name + "_conv1", false);
  out = compute_bn_layer(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn1"));
  out = compute_relu_layer(out, h0 * w0 * c0);

  // 第二个 conv + BN + RELU 结构
  out = compute_conv_layer(out, h0, w0, h1, w1, c1, bottleneck_layer_name + std::string("_conv2"));
  out = compute_bn_layer(out, h1, w1, c1, bottleneck_layer_name + std::string("_bn2"));
  out = compute_relu_layer(out, h1 * w1 * c1);

  // 第三个 conv + BN + RELU 结构
  out = compute_conv_layer(out, h1, w1, h0, w0, c0, bottleneck_layer_name + std::string("_conv3"));
  auto bn_out = compute_bn_layer(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn3"));

  // 定义一个加法函数，用于完成两个张量对位元素的相加
  auto add = [](float* l, float* r, float* out, int len) -> float* {
    for (int i = 0; i < len; i++) {
      out[i] = l[i] + r[i];  // 按元素相加
    }
    return out;
  };

  // 如果结构中存在下采样
  if (down_sample) {
    int h2, w2, c2;
    // 执行卷积层
    auto conv_out = compute_conv_layer(in_data, hi, wi, h2, w2, c2,
                                       bottleneck_layer_name + std::string("_downsample_conv2d"));
    // 执行批量归一化层
    auto short_cut_out = compute_bn_layer(
        conv_out, h2, w2, c2, bottleneck_layer_name + std::string("_downsample_batchnorm"));
    // 残差连接
    add(bn_out, short_cut_out, bn_out, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return compute_relu_layer(bn_out, h2 * w2 * c2);
  } else {
    // 不需要下采样，直接将输入和第三个 conv+bn 层的输出相加
    ho = h0, wo = w0, co = c0;
    add(bn_out, in_data, bn_out, h0 * w0 * c0);
    free(in_data);
    return compute_relu_layer(bn_out, h0 * w0 * c0);
  }
}
