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

// 3_preload 相比于 2_avx2， 优化了模型中权值和参数的加载方式。
// 2_avx2 及之前的版本，模型的权值和参数的加载，是边推理边 malloc 内存边从 txt
// 文件中加载的，这样比较耗时 2_preload 将所有对权值和参数的加载过程，放在了模型推理之前，调用带
// PreLoad 后缀的函数完成。 Preload
// 后缀的函数，仅仅为实现权值的加载，不完成任何计算，因此会看到很多函数都是空的。
// 在推理之前，会调用一次 PreLoadParams() 函数来完成所有层的权值、参数加载，并存放到一个全局变量
// __global_params 中 模型推理时，用到的所有权值和参数，直接从 __global_params 取用即可。

// 全局参数map，用于存储网络预加载的参数
std::map<std::string, void*> __global_params;

// 模板函数，用于从文件加载数据
// 该函数在预加载阶段调用，将 malloc 出来的内存 data 存放在 __global_params
// 中，这个过程模拟权值预加载。
template <typename T>
void* LoadData(const std::string& file_name, int len, bool is_float) {
  T* data = (T*)malloc(len * sizeof(T));     // 分配内存
  FILE* fp = fopen(file_name.c_str(), "r");  // 打开文件
  // 遍历文件中的每一个元素
  for (auto i = 0; i < len; i++) {
    float x = 0;
    auto d = fscanf(fp, "%f", &x);    // 读取文件中的浮点数
    data[i] = is_float ? x : (int)x;  // 根据数据类型存储数据
  }
  fclose(fp);                         // 关闭文件
  __global_params[file_name] = data;  // 将数据存储到全局map中
  return (void*)data;                 // 返回数据指针
}

// 用于预加载卷积层的权重，调用LoadData将卷积层的weight加载到 __global_params 中
float* LoadCon2dWeightPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return (float*)LoadData<float>(file_name, len, true);  // 调用LoadData模板函数加载数据
}

// 真正推理时调用的函数，直接从 __global_params 中读取 weight 所在内存
// 该函数与 LoadCon2dWeightPreLoad 配合，一个完成数据加载到内存，一个完成数据从内存中读取数据
// 这样在真正推理时，就没有了 malloc 操作，仅仅需要从__global_params返回已经 malloc 好的地址
// 大幅增加推理性能，其余层的逻辑类似
float* LoadCon2dWeight(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return (float*)__global_params[file_name];  // 从全局map中获取数据
}

// 用于预加载卷积层的参数
int* LoadCon2dParamPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return (int*)LoadData<int>(file_name, len, false);  // 调用LoadData模板函数加载数据
}

// 用于从全局map中获取卷积层的参数
int* LoadCon2dParam(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return (int*)__global_params[file_name];  // 从全局map中获取数据
}

// 用于ReLU层的计算（预加载时直接返回原图像）
float* ComputeLayerReluPreLoad(float* img, int len) { return img; }

// 用于ReLU层的计算
float* ComputeLayerRelu(float* img, int len) {
  // 遍历图像中的每个像素，应用ReLU激活函数
  for (int i = 0; i < len; i++) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
  return img;  // 返回处理后的图像
}

// 预加载时使用：计算卷积层
float* ComputeLayerConv2dPreLoad(float* img,
                                 int hi,
                                 int wi,
                                 int& ho,
                                 int& wo,
                                 int& co,
                                 const std::string& layer_name,
                                 bool is_free_img = true) {
  auto param = LoadCon2dParamPreLoad(layer_name, 5);  // 加载卷积层的参数
  auto ci = param[0];                                 // 输入通道数
  co = param[1];                                      // 输出通道数
  auto kernel = param[2];                             // 卷积核大小
  auto stride = param[3];                             // 步长
  auto pad = param[4];                                // 填充
  auto weight = LoadCon2dWeightPreLoad(layer_name, co * kernel * kernel * ci);  // 加载卷积层的权重
  return MyConv2dPreLoad(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad);
}

// 推理时使用: 用于计算卷积层
float* ComputeLayerConv2d(float* img,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& layer_name,
                          bool is_free_img = true) {
  auto param = LoadCon2dParam(layer_name, 5);                            // 加载卷积层的参数
  auto ci = param[0];                                                    // 输入通道数
  co = param[1];                                                         // 输出通道数
  auto kernel = param[2];                                                // 卷积核大小
  auto stride = param[3];                                                // 步长
  auto pad = param[4];                                                   // 填充
  auto weight = LoadCon2dWeight(layer_name, co * kernel * kernel * ci);  // 加载卷积层的权重
  if (hi == 224) {
    // 如果输入高度为224，为第一层卷积，调用MyConv2d函数进行卷积计算
    return MyConv2d(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, true, is_free_img);
  } else {
    return MyConv2d(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, false, is_free_img);
  }
}

// 预加载全连接层参数
float* ComputeLayerFCPreLoad(float* img, const std::string& layer_name) {
  std::string weight_file_name =
      "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  std::string bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  LoadData<float>(weight_file_name, 1000 * 2048, true);  // 加载权重
  LoadData<float>(bias_file_name, 1000, true);           // 加载偏置
  return img;
}

// 推理时计算全连接层
float* ComputeLayerFC(float* img, const std::string& layer_name) {
  std::string weight_file_name =
      "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  std::string bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto weight = (float*)__global_params[weight_file_name];  // 从全局参数中获取权重
  auto bias = (float*)__global_params[bias_file_name];      // 从全局参数中获取偏置
  return MyFC(img, weight, bias);  // 调用MyFC函数进行全连接层计算
}

// 预加载批归一化层
float* ComputeLayerBatchNormPreLoad(
    float* in_data, int h, int w, int c, const std::string& layer_name) {
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
  LoadData<float>(weight_file_name, c, true);  // 加载gamma参数
  LoadData<float>(bias_file_name, c, true);    // 加载偏置
  LoadData<float>(mean_file_name, c, true);    // 加载均值
  LoadData<float>(var_file_name, c, true);     // 加载方差
  return in_data;
}

// 批归一化层的计算函数
float* ComputeLayerBatchNorm(float* in_data, int h, int w, int c, const std::string& layer_name) {
  // 构建文件路径，用于加载权重和参数
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";

  // 从全局参数中获取批归一化层的权重和参数
  auto gamma = (float*)__global_params[weight_file_name];
  auto bias = (float*)__global_params[bias_file_name];
  auto mean = (float*)__global_params[mean_file_name];
  auto var = (float*)__global_params[var_file_name];

  // 调用批归一化的实际计算函数
  return MyBatchNorm(in_data, mean, var, gamma, bias, h, w, c);
}

// 最大池化层的预加载函数（空实现，不做任何处理）
float* ComputeLayerMaxPoolPreLoad(float* in_data) { return in_data; }

// 最大池化层的计算函数
float* ComputeLayerMaxPool(float* in_data) { return MyMaxPool(in_data); }

// 平均池化层的预加载函数（空实现，不做任何处理）
float* ComputeLayerAvgPoolPreLoad(float* in_data) { return in_data; }

// 平均池化层的计算函数
float* ComputeLayerAvgPool(float* in_data) { return MyAvgPool(in_data); }

// 瓶颈层的预加载函数
float* ComputeBottleNeckPreLoad(float* in_data,
                                int hi,
                                int wi,
                                int& ho,
                                int& wo,
                                int& co,
                                const std::string& bottleneck_layer_name,
                                bool down_sample) {
  int h0, w0, c0;
  int h1, w1, c1;

  // 计算瓶颈层的各个子层
  auto out = ComputeLayerConv2dPreLoad(in_data, hi, wi, h0, w0, c0,
                                       bottleneck_layer_name + "_conv1", false);
  out = ComputeLayerBatchNormPreLoad(out, h0, w0, c0, bottleneck_layer_name + "_bn1");
  out = ComputeLayerReluPreLoad(out, h0 * w0 * c0);
  out = ComputeLayerConv2dPreLoad(out, h0, w0, h1, w1, c1, bottleneck_layer_name + "_conv2");
  out = ComputeLayerBatchNormPreLoad(out, h1, w1, c1, bottleneck_layer_name + "_bn2");
  out = ComputeLayerReluPreLoad(out, h1 * w1 * c1);
  out = ComputeLayerConv2dPreLoad(out, h1, w1, h0, w0, c0, bottleneck_layer_name + "_conv3");
  auto bn_out = ComputeLayerBatchNormPreLoad(out, h0, w0, c0, bottleneck_layer_name + "_bn3");

  auto Add = [](float* l, float* r, float* out, int len) -> float* {
    // 空实现的累加函数
    return l;
  };

  // 处理下采样
  if (down_sample) {
    int h2, w2, c2;
    auto conv_out = ComputeLayerConv2dPreLoad(in_data, hi, wi, h2, w2, c2,
                                              bottleneck_layer_name + "_downsample_conv2d");
    auto short_cut_out = ComputeLayerBatchNormPreLoad(
        conv_out, h2, w2, c2, bottleneck_layer_name + "_downsample_batchnorm");
    Add(bn_out, short_cut_out, bn_out, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return ComputeLayerReluPreLoad(bn_out, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    Add(bn_out, in_data, bn_out, h0 * w0 * c0);
    free(in_data);
    return ComputeLayerReluPreLoad(bn_out, h0 * w0 * c0);
  }
}

// 瓶颈层的计算函数
float* ComputeBottleNeck(float* in_data,
                         int hi,
                         int wi,
                         int& ho,
                         int& wo,
                         int& co,
                         const std::string& bottleneck_layer_name,
                         bool down_sample) {
  int h0, w0, c0;
  int h1, w1, c1;

  // 计算各个子层
  auto out =
      ComputeLayerConv2d(in_data, hi, wi, h0, w0, c0, bottleneck_layer_name + "_conv1", false);
  out = ComputeLayerBatchNorm(out, h0, w0, c0, bottleneck_layer_name + "_bn1");
  out = ComputeLayerRelu(out, h0 * w0 * c0);
  out = ComputeLayerConv2d(out, h0, w0, h1, w1, c1, bottleneck_layer_name + "_conv2");
  out = ComputeLayerBatchNorm(out, h1, w1, c1, bottleneck_layer_name + "_bn2");
  out = ComputeLayerRelu(out, h1 * w1 * c1);
  out = ComputeLayerConv2d(out, h1, w1, h0, w0, c0, bottleneck_layer_name + "_conv3");
  auto bn_out = ComputeLayerBatchNorm(out, h0, w0, c0, bottleneck_layer_name + "_bn3");

  auto Add = [](float* l, float* r, float* out, int len) -> float* {
    // 执行元素级别的加法操作
    for (int i = 0; i < len; i += 8) {
      out[i + 0] = l[i + 0] + r[i + 0];
      out[i + 1] = l[i + 1] + r[i + 1];
      out[i + 2] = l[i + 2] + r[i + 2];
      out[i + 3] = l[i + 3] + r[i + 3];
      out[i + 4] = l[i + 4] + r[i + 4];
      out[i + 5] = l[i + 5] + r[i + 5];
      out[i + 6] = l[i + 6] + r[i + 6];
      out[i + 7] = l[i + 7] + r[i + 7];
    }
    return out;
  };

  // 处理下采样
  if (down_sample) {
    int h2, w2, c2;
    auto conv_out = ComputeLayerConv2d(in_data, hi, wi, h2, w2, c2,
                                       bottleneck_layer_name + "_downsample_conv2d");
    auto short_cut_out = ComputeLayerBatchNorm(conv_out, h2, w2, c2,
                                               bottleneck_layer_name + "_downsample_batchnorm");
    Add(bn_out, short_cut_out, bn_out, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return ComputeLayerRelu(bn_out, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    Add(bn_out, in_data, bn_out, h0 * w0 * c0);
    free(in_data);
    return ComputeLayerRelu(bn_out, h0 * w0 * c0);
  }
}

// 预加载网络参数的函数
void PreLoadParams() {
  float* img = nullptr;
  int h0, w0, c0;
  int h1, w1, c1;

  // 预加载各层网络参数，以便加快后续计算速度
  img = ComputeLayerConv2dPreLoad(img, 224, 224, h1, w1, c1, "conv1");
  img = ComputeLayerBatchNormPreLoad(img, h1, w1, c1, "bn1");
  img = ComputeLayerReluPreLoad(img, h1 * w1 * c1);
  img = ComputeLayerMaxPoolPreLoad(img);
  // layer1
  img = ComputeBottleNeckPreLoad(img, 56, 56, h1, w1, c1, "layer1_bottleneck0", true);
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer1_bottleneck1", false);
  img = ComputeBottleNeckPreLoad(img, h0, w0, h1, w1, c1, "layer1_bottleneck2", false);
  // layer2
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer2_bottleneck0", true);
  img = ComputeBottleNeckPreLoad(img, h0, w0, h1, w1, c1, "layer2_bottleneck1", false);
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer2_bottleneck2", false);
  img = ComputeBottleNeckPreLoad(img, h0, w0, h1, w1, c1, "layer2_bottleneck3", false);
  // layer3
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer3_bottleneck0", true);
  img = ComputeBottleNeckPreLoad(img, h0, w0, h1, w1, c1, "layer3_bottleneck1", false);
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer3_bottleneck2", false);
  img = ComputeBottleNeckPreLoad(img, h0, w0, h1, w1, c1, "layer3_bottleneck3", false);
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer3_bottleneck4", false);
  img = ComputeBottleNeckPreLoad(img, h0, w0, h1, w1, c1, "layer3_bottleneck5", false);
  // layer4
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer4_bottleneck0", true);
  img = ComputeBottleNeckPreLoad(img, h0, w0, h1, w1, c1, "layer4_bottleneck1", false);
  img = ComputeBottleNeckPreLoad(img, h1, w1, h0, w0, c0, "layer4_bottleneck2", false);
  // avg pool
  img = ComputeLayerAvgPoolPreLoad(img);
  // Linear
  img = ComputeLayerFCPreLoad(img, "fc");
  return;
}
