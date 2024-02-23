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

// optimize by pre-load params of networks
std::map<std::string, void*> __global_params;

template <typename T>
void* LoadData(const std::string& file_name, int len, bool is_float) {
  T* data = (T*)malloc(len * sizeof(T));
  FILE* fp = fopen(file_name.c_str(), "r");
  // std::cout << "file_name = " << file_name << ", fp = " << fp << std::endl;
  for (auto i = 0; i < len; i++) {
    float x = 0;
    auto d = fscanf(fp, "%f", &x);
    data[i] = is_float ? x : (int)x;
  }
  fclose(fp);
  __global_params[file_name] = data;
  return (void*)data;
}

float* LoadCon2dWeightPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return (float*)LoadData<float>(file_name, len, true);
}

float* LoadCon2dWeight(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return (float*)__global_params[file_name];
}

int* LoadCon2dParamPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return (int*)LoadData<int>(file_name, len, false);
}

int* LoadCon2dParam(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return (int*)__global_params[file_name];
}

float* ComputeLayerReluPreLoad(float* img, int len) { return img; }

float* ComputeLayerRelu(float* img, int len) {
  for (int i = 0; i < len; i++) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
  return img;
}

float* ComputeLayerConv2dPreLoad(float* img,
                                 int hi,
                                 int wi,
                                 int& ho,
                                 int& wo,
                                 int& co,
                                 const std::string& layer_name,
                                 bool is_free_img = true) {
  auto param = LoadCon2dParamPreLoad(layer_name, 5);
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeightPreLoad(layer_name, co * kernel * kernel * ci);
  if (hi == 224) {
    return MyConv2dPreLoad(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad);
  } else {
    return MyConv2dPreLoad(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad);
  }
}

float* ComputeLayerConv2d(float* img,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& layer_name,
                          bool is_free_img = true) {
  auto param = LoadCon2dParam(layer_name, 5);
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeight(layer_name, co * kernel * kernel * ci);
  if (hi == 224) {
    return MyConv2d(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, true, is_free_img);
  } else {
    return MyConv2d(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, false, is_free_img);
  }
}

float* ComputeLayerFCPreLoad(float* img, const std::string& layer_name) {
  std::string weight_file_name =
      "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  std::string bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  LoadData<float>(weight_file_name, 1000 * 2048, true);
  LoadData<float>(bias_file_name, 1000, true);
  return img;
}

float* ComputeLayerFC(float* img, const std::string& layer_name) {
  std::string weight_file_name =
      "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  std::string bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto weight = (float*)__global_params[weight_file_name];
  auto bias = (float*)__global_params[bias_file_name];
  return MyFC(img, weight, bias);
}

float* ComputeLayerBatchNormPreLoad(
    float* in_data, int h, int w, int c, const std::string& layer_name) {
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
  LoadData<float>(weight_file_name, c, true);
  LoadData<float>(bias_file_name, c, true);
  LoadData<float>(mean_file_name, c, true);
  LoadData<float>(var_file_name, c, true);
  return in_data;
}

float* ComputeLayerBatchNorm(float* in_data, int h, int w, int c, const std::string& layer_name) {
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
  auto gamma = (float*)__global_params[weight_file_name];
  auto bias = (float*)__global_params[bias_file_name];
  auto mean = (float*)__global_params[mean_file_name];
  auto var = (float*)__global_params[var_file_name];
  return MyBatchNorm(in_data, mean, var, gamma, bias, h, w, c);
}

float* ComputeLayerMaxPoolPreLoad(float* in_data) { return in_data; }

float* ComputeLayerMaxPool(float* in_data) { return MyMaxPool(in_data); }

float* ComputeLayerAvgPoolPreLoad(float* in_data) { return in_data; }

float* ComputeLayerAvgPool(float* in_data) { return MyAvgPool(in_data); }

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
  auto out = ComputeLayerConv2dPreLoad(in_data, hi, wi, h0, w0, c0,
                                       bottleneck_layer_name + "_conv1", false);
  out = ComputeLayerBatchNormPreLoad(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn1"));
  out = ComputeLayerReluPreLoad(out, h0 * w0 * c0);

  out = ComputeLayerConv2dPreLoad(out, h0, w0, h1, w1, c1,
                                  bottleneck_layer_name + std::string("_conv2"));
  out = ComputeLayerBatchNormPreLoad(out, h1, w1, c1, bottleneck_layer_name + std::string("_bn2"));
  out = ComputeLayerReluPreLoad(out, h1 * w1 * c1);

  out = ComputeLayerConv2dPreLoad(out, h1, w1, h0, w0, c0,
                                  bottleneck_layer_name + std::string("_conv3"));
  auto bn_out =
      ComputeLayerBatchNormPreLoad(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn3"));

  auto Add = [](float* l, float* r, float* out, int len) -> float* { return l; };

  if (down_sample) {
    int h2, w2, c2;
    auto conv_out = ComputeLayerConv2dPreLoad(
        in_data, hi, wi, h2, w2, c2, bottleneck_layer_name + std::string("_downsample_conv2d"));
    auto short_cut_out = ComputeLayerBatchNormPreLoad(
        conv_out, h2, w2, c2, bottleneck_layer_name + std::string("_downsample_batchnorm"));
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
  auto out =
      ComputeLayerConv2d(in_data, hi, wi, h0, w0, c0, bottleneck_layer_name + "_conv1", false);
  out = ComputeLayerBatchNorm(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn1"));
  out = ComputeLayerRelu(out, h0 * w0 * c0);

  out = ComputeLayerConv2d(out, h0, w0, h1, w1, c1, bottleneck_layer_name + std::string("_conv2"));
  out = ComputeLayerBatchNorm(out, h1, w1, c1, bottleneck_layer_name + std::string("_bn2"));
  out = ComputeLayerRelu(out, h1 * w1 * c1);

  out = ComputeLayerConv2d(out, h1, w1, h0, w0, c0, bottleneck_layer_name + std::string("_conv3"));
  auto bn_out = ComputeLayerBatchNorm(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn3"));

  auto Add = [](float* l, float* r, float* out, int len) -> float* {
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

  if (down_sample) {
    int h2, w2, c2;
    auto conv_out = ComputeLayerConv2d(in_data, hi, wi, h2, w2, c2,
                                       bottleneck_layer_name + std::string("_downsample_conv2d"));
    auto short_cut_out = ComputeLayerBatchNorm(
        conv_out, h2, w2, c2, bottleneck_layer_name + std::string("_downsample_batchnorm"));
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

void PreLoadParams() {
  float* img = nullptr;
  int h0, w0, c0;
  int h1, w1, c1;
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
