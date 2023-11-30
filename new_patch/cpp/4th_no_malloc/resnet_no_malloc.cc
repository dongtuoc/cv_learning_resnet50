#include "resnet_no_malloc.h"

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

// unlikely to predict un-commonly-used branches
#define unlikely(x) __builtin_expect(!!(x), 0)

// optimize by pre-load params of networks
void* __global_weight[MAX_MEM_NUM] = {nullptr};

int put_cnt = 0;
int out_cnt = 0;

template <typename T>
void* LoadData(const std::string& file_name, int len, bool is_float) {
  T* data = (T*)malloc(len * sizeof(T));
  FILE* fp = fopen(file_name.c_str(), "r");
  // std::cout << "file_name = " << file_name << ", fp = " << fp << std::endl;
  for (auto i = 0; i < len; i++) {
    float x = 0;
    fscanf(fp, "%f", &x);
    data[i] = is_float ? x : (int)x;
  }
  fclose(fp);
  __global_weight[put_cnt++] = data;
  return (void*)data;
}

float* LoadCon2dWeightPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return (float*)LoadData<float>(file_name, len, true);
}

float* LoadCon2dWeight() { return (float*)__global_weight[out_cnt++]; }

int* LoadCon2dParamPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return (int*)LoadData<int>(file_name, len, false);
}

int* LoadCon2dParam() { return (int*)__global_weight[out_cnt++]; }

void ComputeLayerReluPreLoad(void* img_in, int len) { return; }

void ComputeLayerRelu(void* img_in, int len) {
  float* img = (float*)img_in;
  for (int i = 0; i < len; i++) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}

void ComputeLayerConv2dPreLoad(void* img_in,
                               void* img_out,
                               int hi,
                               int wi,
                               int& ho,
                               int& wo,
                               int& co,
                               const std::string& layer_name) {
  auto param = LoadCon2dParamPreLoad(layer_name, 5);
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeightPreLoad(layer_name, co * kernel * kernel * ci);
  if (hi == 224) {
    return MyConv2dPreLoad(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad);
  } else {
    return MyConv2dPreLoad(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad);
  }
}

void ComputeLayerConv2d(void* img_in, void* img_out, int hi, int wi, int& ho, int& wo, int& co) {
  auto param = LoadCon2dParam();
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeight();
  if (hi == 224) {
    return MyConv2d(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, true);
  } else {
    return MyConv2d(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, false);
  }
}

void ComputeLayerFCPreLoad(void* img_in, void* img_out, const std::string& layer_name) {
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  LoadData<float>(weight_file_name, 1000 * 2048, true);
  LoadData<float>(bias_file_name, 1000, true);
  return;
}

void ComputeLayerFC(void* img_in, void* img_out) {
  auto weight = (float*)__global_weight[out_cnt++];
  auto bias = (float*)__global_weight[out_cnt++];
  return MyFC(img_in, img_out, weight, bias);
}

void ComputeLayerBatchNormPreLoad(
    void* in_data, void* out_data, int h, int w, int c, const std::string& layer_name) {
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
  LoadData<float>(weight_file_name, c, true);
  LoadData<float>(bias_file_name, c, true);
  LoadData<float>(mean_file_name, c, true);
  LoadData<float>(var_file_name, c, true);
  return;
}

void ComputeLayerBatchNorm(void* in_data, void* out_data, int h, int w, int c) {
  auto gamma = (float*)__global_weight[out_cnt++];
  auto bias = (float*)__global_weight[out_cnt++];
  auto mean = (float*)__global_weight[out_cnt++];
  auto var = (float*)__global_weight[out_cnt++];
  return MyBatchNorm(in_data, out_data, mean, var, gamma, bias, h, w, c);
}

void ComputeLayerMaxPoolPreLoad(void* in_data, void* out_data) { return; }

void ComputeLayerMaxPool(void* in_data, void* out_data) { return MyMaxPool(in_data, out_data); }

void ComputeLayerAvgPoolPreLoad(void* in_data, void* out_data) { return; }

void ComputeLayerAvgPool(void* in_data, void* out_data) { return MyAvgPool(in_data, out_data); }

void AddPreLoad(float* l, float* r, float* out, int len) { return; }

void Add(float* l, float* r, float* out, int len) {
#if 1
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
#else
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
#endif
  return;
}

void ComputeBottleNeckPreLoad(void* in_data,
                              void* out_data,
                              void* temp_data,
                              int hi,
                              int wi,
                              int& ho,
                              int& wo,
                              int& co,
                              const std::string& bottleneck_layer_name,
                              bool down_sample) {
  int h0, w0, c0;
  int h1, w1, c1;

  ComputeLayerConv2dPreLoad(in_data, out_data, hi, wi, h0, w0, c0,
                            bottleneck_layer_name + "_conv1");
  ComputeLayerBatchNormPreLoad(out_data, temp_data, h0, w0, c0,
                               bottleneck_layer_name + std::string("_bn1"));
  ComputeLayerReluPreLoad(temp_data, h0 * w0 * c0);

  ComputeLayerConv2dPreLoad(temp_data, out_data, h0, w0, h1, w1, c1,
                            bottleneck_layer_name + std::string("_conv2"));
  ComputeLayerBatchNormPreLoad(out_data, temp_data, h1, w1, c1,
                               bottleneck_layer_name + std::string("_bn2"));
  ComputeLayerReluPreLoad(temp_data, h1 * w1 * c1);

  ComputeLayerConv2dPreLoad(temp_data, out_data, h1, w1, h0, w0, c0,
                            bottleneck_layer_name + std::string("_conv3"));
  ComputeLayerBatchNormPreLoad(out_data, temp_data, h0, w0, c0,
                               bottleneck_layer_name + std::string("_bn3"));
  auto bn_out = temp_data;

  if (unlikely(down_sample)) {
    int h2, w2, c2;
    ComputeLayerConv2dPreLoad(in_data, out_data, hi, wi, h2, w2, c2,
                              bottleneck_layer_name + std::string("_downsample_conv2d"));
    ComputeLayerBatchNormPreLoad(out_data, in_data, h2, w2, c2,
                                 bottleneck_layer_name + std::string("_downsample_batchnorm"));
    auto short_cut_out = in_data;
    AddPreLoad((float*)bn_out, (float*)short_cut_out, (float*)out_data, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return ComputeLayerReluPreLoad(out_data, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    AddPreLoad((float*)bn_out, (float*)in_data, (float*)out_data, h0 * w0 * c0);
    return ComputeLayerReluPreLoad(out_data, h0 * w0 * c0);
  }
}

void ComputeBottleNeck(void* in_data,
                       void* out_data,
                       void* temp_data,
                       int hi,
                       int wi,
                       int& ho,
                       int& wo,
                       int& co,
                       bool down_sample) {
  int h0, w0, c0;
  int h1, w1, c1;

  ComputeLayerConv2d(in_data, out_data, hi, wi, h0, w0, c0);
  ComputeLayerBatchNorm(out_data, temp_data, h0, w0, c0);
  ComputeLayerRelu(temp_data, h0 * w0 * c0);

  ComputeLayerConv2d(temp_data, out_data, h0, w0, h1, w1, c1);
  ComputeLayerBatchNorm(out_data, temp_data, h1, w1, c1);
  ComputeLayerRelu(temp_data, h1 * w1 * c1);

  ComputeLayerConv2d(temp_data, out_data, h1, w1, h0, w0, c0);
  ComputeLayerBatchNorm(out_data, temp_data, h0, w0, c0);
  auto bn_out = temp_data;

  if (unlikely(down_sample)) {
    int h2, w2, c2;
    ComputeLayerConv2d(in_data, out_data, hi, wi, h2, w2, c2);
    ComputeLayerBatchNorm(out_data, in_data, h2, w2, c2);
    auto short_cut_out = in_data;
    Add((float*)bn_out, (float*)short_cut_out, (float*)out_data, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return ComputeLayerRelu(out_data, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    Add((float*)bn_out, (float*)in_data, (float*)out_data, h0 * w0 * c0);
    return ComputeLayerRelu(out_data, h0 * w0 * c0);
  }
}

void PreLoadParams() {
  float* img0 = nullptr;
  float* img1 = nullptr;
  float* img2 = nullptr;
  int h0, w0, c0;
  int h1, w1, c1;
  ComputeLayerConv2dPreLoad(img0, img1, 224, 224, h1, w1, c1, "conv1");
  ComputeLayerBatchNormPreLoad(img1, img0, h1, w1, c1, "bn1");
  ComputeLayerReluPreLoad(img0, h1 * w1 * c1);
  ComputeLayerMaxPoolPreLoad(img0, img1);
  // layer1
  ComputeBottleNeckPreLoad(img1, img0, img2, 56, 56, h1, w1, c1, "layer1_bottleneck0", true);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer1_bottleneck1", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer1_bottleneck2", false);
  // layer2
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck0", true);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck1", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck2", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck3", false);
  // layer3
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck0", true);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck1", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck2", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck3", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck4", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck5", false);
  // layer4
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck0", true);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer4_bottleneck1", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck2", false);
  // avg pool
  ComputeLayerAvgPoolPreLoad(img1, img0);
  // Linear
  ComputeLayerFCPreLoad(img0, img1, "fc");
  return;
}
