#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#define DEBUG_SHOW (0)

static inline void show(float* out, int len) {
  printf("out : %f\n", out[0]);
  printf("out : %f\n", out[1]);
  printf("out : %f\n", out[2]);
  printf("out : %f\n", out[3]);
  printf("out 63: %f\n\n", out[len - 1]);
  printf("out 64: %f\n", out[len]);
  exit(0);
}

template <bool is_first = false>
static float* my_conv2d(float* img,
                        float* weight,
                        int hi,
                        int wi,
                        int& ho,
                        int& wo,
                        int ci,
                        int co,
                        int kernel,
                        int stride,
                        int pad,
                        bool is_free_img = true) {
#if DEBUG_SHOW
  printf("conv in: (%d, %d, %d)\n", hi, wi, ci);
#endif
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  float* out = (float*)malloc(ho * wo * co * sizeof(float));

  for (int co_idx = 0; co_idx < co; co_idx++) {
    for (int ho_idx = 0; ho_idx < ho; ho_idx++) {
      const int in_h_origin = ho_idx * stride - pad;
      for (int wo_idx = 0; wo_idx < wo; wo_idx++) {
        const int in_w_origin = wo_idx * stride - pad;
        const int filter_h_start = std::max(0, -in_h_origin);
        const int filter_w_start = std::max(0, -in_w_origin);
        const int filter_h_end = std::min(kernel, hi - in_h_origin);
        const int filter_w_end = std::min(kernel, wi - in_w_origin);
        register float acc = 0;
        if (is_first) {
          for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
            const int hi_index = in_h_origin + kh_idx;
            for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
              const int wi_index = in_w_origin + kw_idx;
              for (int ci_ = 0; ci_ < 3; ci_++) {
                auto in_data = img[hi_index * 224 * 3 + wi_index * 3 + ci_];
                auto weight_data = weight[co_idx * 49 * 3 + kh_idx * 7 * 3 + kw_idx * 3 + ci_];
                acc += in_data * weight_data;
              }
            }
          }
          out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;
        } else {
          // use avx2 vec inst to optimize Mul-add operation
          const int vec_size = 8;
          for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
            const register int hi_index = in_h_origin + kh_idx;
            for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
              const register int wi_index = in_w_origin + kw_idx;
              // Load input and weight data into vectors
              __m256 in_vec, weight_vec;
              for (int ci_ = 0; ci_ < ci; ci_ += vec_size) {
                in_vec = _mm256_loadu_ps(&img[hi_index * wi * ci + wi_index * ci + ci_]);
                weight_vec = _mm256_loadu_ps(&weight[co_idx * kernel * kernel * ci +
                                                     kh_idx * kernel * ci + kw_idx * ci + ci_]);
                in_vec = _mm256_mul_ps(in_vec, weight_vec);
                // Add the elements of the accumulator vector and store the result
                float* acc_ptr = (float*)&in_vec;
                for (int i = 0; i < vec_size; i++) {
                  acc += acc_ptr[i];
                }
              }
            }
          }
          out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;
        }
      }
    }
  }

  if (is_free_img) {
    free(img);
  }
  free(weight);
#if DEBUG_SHOW
  printf("conv out: (%d, %d, %d)\n", ho, wo, co);
#endif
  return out;
}

static float* my_fc(float* img, float* weight, float* bias) {
#if DEBUG_SHOW
  printf("fc in: (1000, 2048)\n");
  printf("fc out: (1000)\n");
#endif
  float* out = (float*)malloc(1000 * sizeof(float));
  for (int i = 0; i < 1000; i++) {
    float sum_x = float(0);
    for (int j = 0; j < 2048; j++) {
      auto l = img[j];
      auto r = weight[i * 2048 + j];
      sum_x += l * r;
    }
    out[i] = sum_x + bias[i];
  }
  free(img);
  free(weight);
  free(bias);
  return out;
}

static float* my_max_pool(float* img) {
  const auto hi = 112;
  const auto wi = 112;
  const auto channel = 64;
  const auto pad = 1;
  const auto stride = 2;
  const auto kernel = 3;
  const auto ho = (hi + 2 * pad - kernel) / stride + 1;
  const auto wo = (wi + 2 * pad - kernel) / stride + 1;
#if DEBUG_SHOW
  printf("maxpool in: (%d, %d, %d)\n", hi, wi, channel);
#endif
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));

  for (auto c_ = 0; c_ < channel; c_++) {
    for (auto ho_idx = 0; ho_idx < ho; ho_idx++) {
      int in_h_origin = ho_idx * stride - pad;
      for (auto wo_idx = 0; wo_idx < wo; wo_idx++) {
        int in_w_origin = wo_idx * stride - pad;
        auto filter_h_start = std::max(0, -in_h_origin);
        auto filter_w_start = std::max(0, -in_w_origin);
        auto filter_h_end = std::min(kernel, hi - in_h_origin);
        auto filter_w_end = std::min(kernel, wi - in_w_origin);
        float max_x = float(0);
        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          auto hi_index = in_h_origin + kh_idx;
          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            auto wi_index = in_w_origin + kw_idx;
            auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
            max_x = std::max(in_data, max_x);
          }
        }
        out[ho_idx * wo * channel + wo_idx * channel + c_] = max_x;
      }
    }
  }
  free(img);
#if DEBUG_SHOW
  printf("maxpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
  return out;
}

static float* my_avg_pool(float* img) {
  const auto hi = 7;
  const auto wi = 7;
  const auto channel = 2048;
  const auto pad = 0;
  const auto stride = 1;
  const auto kernel = 7;
  const auto ho = (hi + 2 * pad - kernel) / stride + 1;
  const auto wo = (wi + 2 * pad - kernel) / stride + 1;
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));
#if DEBUG_SHOW
  printf("avgpool in: (%d, %d, %d)\n", hi, wi, channel);
#endif

  for (auto c_ = 0; c_ < channel; c_++) {
    for (auto ho_idx = 0; ho_idx < ho; ho_idx++) {
      int in_h_origin = ho_idx * stride - pad;
      for (auto wo_idx = 0; wo_idx < wo; wo_idx++) {
        int in_w_origin = wo_idx * stride - pad;
        auto filter_h_start = std::max(0, -in_h_origin);
        auto filter_w_start = std::max(0, -in_w_origin);
        auto filter_h_end = std::min(kernel, hi - in_h_origin);
        auto filter_w_end = std::min(kernel, wi - in_w_origin);
        float sum = float(0);
        int k_size = (filter_h_end - filter_h_start) * (filter_w_end - filter_w_start);
        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          auto hi_index = in_h_origin + kh_idx;
          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            auto wi_index = in_w_origin + kw_idx;
            auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
            sum += in_data;
          }
        }
        out[ho_idx * wo * channel + wo_idx * channel + c_] = sum / k_size;
      }
    }
  }
  free(img);
#if DEBUG_SHOW
  printf("avgpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
  return out;
}

static float* my_bn(
    float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c) {
#if DEBUG_SHOW
  printf("bn in : (%d, %d, %d)\n", h, w, c);
#endif
  float* out = (float*)malloc(h * w * c * sizeof(float));
  for (auto c_ = 0; c_ < c; c_++) {
    auto m = mean[c_];
    auto v = var[c_];
    auto gm = gamma[c_];
    auto bi = bias[c_];
    for (auto hw = 0; hw < h * w; hw++) {
      auto data = img[hw * c + c_];
      auto data_ = (data - m) / sqrt(v + 1e-5);
      data_ = data_ * gm + bi;
      out[hw * c + c_] = data_;
    }
  }
  free(img);
  free(mean);
  free(var);
  free(gamma);
  free(bias);

#if DEBUG_SHOW
  printf("bn out: (%d, %d, %d)\n", h, w, c);
#endif
  return out;
}

static float* compute_relu_layer(float* img, int len) {
#if DEBUG_SHOW
  printf("-- compute relu with %d\n", len);
#endif
  for (int i = 0; i < len; i++) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
  return img;
}

template <typename T>
T* load_data_from_file(const std::string& file_name, int len, bool is_float) {
  T* data = (T*)malloc(len * sizeof(T));
  FILE* fp = fopen(file_name.c_str(), "r");
  // std::cout << "file_name = " << file_name << ", fp = " << fp << std::endl;
  for (auto i = 0; i < len; i++) {
    float x = 0;
    auto d = fscanf(fp, "%f", &x);
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

static float* compute_conv_layer(float* img,
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
  if (hi == 224) {
    return my_conv2d<true>(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, is_free_img);
  } else {
    return my_conv2d(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, is_free_img);
  }
}

static float* compute_fc_layer(float* img, const std::string& layer_name) {
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

static float* compute_bn_layer(float* in_data, int h, int w, int c, const std::string& layer_name) {
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

static float* compute_maxpool_layer(float* in_data) {
#if DEBUG_SHOW
  std::cout << "-- compute maxpool" << std::endl;
#endif
  return my_max_pool(in_data);
}

static float* compute_avgpool_layer(float* in_data) {
#if DEBUG_SHOW
  std::cout << "-- compute avgpool" << std::endl;
#endif
  return my_avg_pool(in_data);
}

static float* compute_bottleneck(float* in_data,
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
