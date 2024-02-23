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

template <typename T>
static T* load_data_from_file(const std::string& file_name, int len, bool is_float) {
  T* data = (T*)malloc(len * sizeof(T));
  FILE* fp = fopen(file_name.c_str(), "r");
  // std::cout << "file_name = " << file_name << ", fp = " << fp << std::endl;
  for (auto i = 0; i < len; i++) {
    float x = 0;
    fscanf(fp, "%f", &x);
    data[i] = is_float ? x : (int)x;
  }
  fclose(fp);
  return data;
}

static float* load_conv_weight(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return load_data_from_file<float>(file_name, len, true);
}

static int* load_conv_param(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return load_data_from_file<int>(file_name, len, false);
}

float* compute_relu_layer(float* img, int len) {
#if DEBUG_SHOW
  printf("-- compute relu with %d\n", len);
#endif
  for (int i = 0; i < len; i++) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
  return img;
}

float* compute_conv_layer(float* img,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& layer_name,
                          bool is_free_img = true) {
#if DEBUG_SHOW
  std::cout << "-- compute " << layer_name << std::endl;
#endif
  auto param = load_conv_param(layer_name, 5);
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = load_conv_weight(layer_name, co * kernel * kernel * ci);
  return my_conv2d(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, is_free_img);
}

float* compute_fc_layer(float* img, const std::string& layer_name) {
#if DEBUG_SHOW
  std::cout << "-- compute " << layer_name << std::endl;
#endif
  std::string weight_file_name =
      "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  std::string bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto weight = load_data_from_file<float>(weight_file_name, 1000 * 2048, true);
  auto bias = load_data_from_file<float>(bias_file_name, 1000, true);
  return my_fc(img, weight, bias);
}

float* compute_bn_layer(float* in_data, int h, int w, int c, const std::string& layer_name) {
#if DEBUG_SHOW
  std::cout << "-- compute " << layer_name << std::endl;
#endif
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
  auto gamma = load_data_from_file<float>(weight_file_name, c, true);
  auto bias = load_data_from_file<float>(bias_file_name, c, true);
  auto mean = load_data_from_file<float>(mean_file_name, c, true);
  auto var = load_data_from_file<float>(var_file_name, c, true);
  return my_bn(in_data, mean, var, gamma, bias, h, w, c);
}

float* compute_maxpool_layer(float* in_data) {
#if DEBUG_SHOW
  std::cout << "-- compute maxpool" << std::endl;
#endif
  return my_max_pool(in_data);
}

float* compute_avgpool_layer(float* in_data) {
#if DEBUG_SHOW
  std::cout << "-- compute avgpool" << std::endl;
#endif
  return my_avg_pool(in_data);
}

float* compute_bottleneck(float* in_data,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& bottleneck_layer_name,
                          bool down_sample) {
#if DEBUG_SHOW
  std::cout << "\n\n-- compute " << bottleneck_layer_name << std::endl;
#endif
  int h0, w0, c0;
  int h1, w1, c1;
  auto out =
      compute_conv_layer(in_data, hi, wi, h0, w0, c0, bottleneck_layer_name + "_conv1", false);
  out = compute_bn_layer(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn1"));
  out = compute_relu_layer(out, h0 * w0 * c0);

  out = compute_conv_layer(out, h0, w0, h1, w1, c1, bottleneck_layer_name + std::string("_conv2"));
  out = compute_bn_layer(out, h1, w1, c1, bottleneck_layer_name + std::string("_bn2"));
  out = compute_relu_layer(out, h1 * w1 * c1);

  out = compute_conv_layer(out, h1, w1, h0, w0, c0, bottleneck_layer_name + std::string("_conv3"));
  auto bn_out = compute_bn_layer(out, h0, w0, c0, bottleneck_layer_name + std::string("_bn3"));

  auto add = [](float* l, float* r, float* out, int len) -> float* {
    for (int i = 0; i < len; i++) {
      out[i] = l[i] + r[i];
    }
    return out;
  };

  if (down_sample) {
    int h2, w2, c2;
    auto conv_out = compute_conv_layer(in_data, hi, wi, h2, w2, c2,
                                       bottleneck_layer_name + std::string("_downsample_conv2d"));
    auto short_cut_out = compute_bn_layer(
        conv_out, h2, w2, c2, bottleneck_layer_name + std::string("_downsample_batchnorm"));
    add(bn_out, short_cut_out, bn_out, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return compute_relu_layer(bn_out, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    add(bn_out, in_data, bn_out, h0 * w0 * c0);
    free(in_data);
    return compute_relu_layer(bn_out, h0 * w0 * c0);
  }
}
