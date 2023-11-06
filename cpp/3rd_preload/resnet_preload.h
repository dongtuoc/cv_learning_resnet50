#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// set DEBUG_SHOW to 1 to enable printf function
#define DEBUG_SHOW (0)

template <bool First, bool PRE_LOAD_PARAM>
static float* MyConv2d(float* img,
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
  if (PRE_LOAD_PARAM) return img;

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
        if (First) {
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
#if DEBUG_SHOW
  printf("conv out: (%d, %d, %d)\n", ho, wo, co);
#endif
  return out;
}

template <bool PRE_LOAD_PARAM>
static float* MyFC(float* img, float* weight, float* bias) {
#if DEBUG_SHOW
  printf("fc in: (1000, 2048)\n");
  printf("fc out: (1000)\n");
#endif
  if (PRE_LOAD_PARAM) return img;
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
  return out;
}

template <bool PRE_LOAD_PARAM>
static float* MyMaxPool(float* img) {
  if (PRE_LOAD_PARAM) return img;
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

template <bool PRE_LOAD_PARAM>
static float* MyAvgPool(float* img) {
  if (PRE_LOAD_PARAM) return img;
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

template <bool PRE_LOAD_PARAM>
static float* MyBatchNorm(
    float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c) {
#if DEBUG_SHOW
  printf("bn in : (%d, %d, %d)\n", h, w, c);
#endif
  if (PRE_LOAD_PARAM) return img;
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

#if DEBUG_SHOW
  printf("bn out: (%d, %d, %d)\n", h, w, c);
#endif
  return out;
}

template <bool PRE_LOAD_PARAM>
static float* ComputeLayerRelu(float* img, int len) {
#if DEBUG_SHOW
  printf("-- compute relu with %d\n", len);
#endif
  if (PRE_LOAD_PARAM) {
    return img;
  } else {
    for (int i = 0; i < len; i++) {
      img[i] = img[i] > 0 ? img[i] : 0;
    }
    return img;
  }
}

// optimize by pre-load params of networks
static std::map<std::string, void*> __global_params;

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

template <bool PRE_LOAD_PARAM>
static float* LoadCon2dWeight(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  if (PRE_LOAD_PARAM) {
    return (float*)LoadData<float>(file_name, len, true);
  } else {
    return (float*)__global_params[file_name];
  }
}

template <bool PRE_LOAD_PARAM>
static int* LoadCon2dParam(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  if (PRE_LOAD_PARAM) {
    return (int*)LoadData<int>(file_name, len, false);
  } else {
    return (int*)__global_params[file_name];
  }
}

template <bool PRE_LOAD_PARAM>
static float* ComputeLayerConv2d(float* img,
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
  auto param = LoadCon2dParam<PRE_LOAD_PARAM>(layer_name, 5);
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeight<PRE_LOAD_PARAM>(layer_name, co * kernel * kernel * ci);
  if (hi == 224) {
    return MyConv2d<true, PRE_LOAD_PARAM>(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad,
                                          is_free_img);
  } else {
    return MyConv2d<false, PRE_LOAD_PARAM>(img, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad,
                                           is_free_img);
  }
}

template <bool PRE_LOAD_PARAM>
static float* ComputeLayerFC(float* img, const std::string& layer_name) {
#if DEBUG_SHOW
  std::cout << "-- compute " << layer_name << std::endl;
#endif
  std::string weight_file_name =
      "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  std::string bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  if (PRE_LOAD_PARAM) {
    LoadData<float>(weight_file_name, 1000 * 2048, true);
    LoadData<float>(bias_file_name, 1000, true);
    return img;
  } else {
    auto weight = (float*)__global_params[weight_file_name];
    auto bias = (float*)__global_params[bias_file_name];
    return MyFC<PRE_LOAD_PARAM>(img, weight, bias);
  }
}

template <bool PRE_LOAD_PARAM>
static float* ComputeLayerBatchNorm(
    float* in_data, int h, int w, int c, const std::string& layer_name) {
#if DEBUG_SHOW
  std::cout << "-- compute " << layer_name << std::endl;
#endif
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
  if (PRE_LOAD_PARAM) {
    LoadData<float>(weight_file_name, c, true);
    LoadData<float>(bias_file_name, c, true);
    LoadData<float>(mean_file_name, c, true);
    LoadData<float>(var_file_name, c, true);
    return in_data;
  } else {
    auto gamma = (float*)__global_params[weight_file_name];
    auto bias = (float*)__global_params[bias_file_name];
    auto mean = (float*)__global_params[mean_file_name];
    auto var = (float*)__global_params[var_file_name];
    return MyBatchNorm<PRE_LOAD_PARAM>(in_data, mean, var, gamma, bias, h, w, c);
  }
}

template <bool PRE_LOAD_PARAM>
static float* ComputeLayerMaxPool(float* in_data) {
#if DEBUG_SHOW
  std::cout << "-- compute maxpool" << std::endl;
#endif
  if (PRE_LOAD_PARAM) {
    return in_data;
  } else {
    return MyMaxPool<PRE_LOAD_PARAM>(in_data);
  }
}

template <bool PRE_LOAD_PARAM>
static float* ComputeLayerAvgPool(float* in_data) {
#if DEBUG_SHOW
  std::cout << "-- compute avgpool" << std::endl;
#endif
  if (PRE_LOAD_PARAM) {
    return in_data;
  } else {
    return MyAvgPool<PRE_LOAD_PARAM>(in_data);
  }
}

template <bool PRE_LOAD_PARAM>
static float* ComputeBottleNeck(float* in_data,
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
  auto out = ComputeLayerConv2d<PRE_LOAD_PARAM>(in_data, hi, wi, h0, w0, c0,
                                                bottleneck_layer_name + "_conv1", false);
  out = ComputeLayerBatchNorm<PRE_LOAD_PARAM>(out, h0, w0, c0,
                                              bottleneck_layer_name + std::string("_bn1"));
  out = ComputeLayerRelu<PRE_LOAD_PARAM>(out, h0 * w0 * c0);

  out = ComputeLayerConv2d<PRE_LOAD_PARAM>(out, h0, w0, h1, w1, c1,
                                           bottleneck_layer_name + std::string("_conv2"));
  out = ComputeLayerBatchNorm<PRE_LOAD_PARAM>(out, h1, w1, c1,
                                              bottleneck_layer_name + std::string("_bn2"));
  out = ComputeLayerRelu<PRE_LOAD_PARAM>(out, h1 * w1 * c1);

  out = ComputeLayerConv2d<PRE_LOAD_PARAM>(out, h1, w1, h0, w0, c0,
                                           bottleneck_layer_name + std::string("_conv3"));
  auto bn_out = ComputeLayerBatchNorm<PRE_LOAD_PARAM>(out, h0, w0, c0,
                                                      bottleneck_layer_name + std::string("_bn3"));

  auto Add = [](float* l, float* r, float* out, int len) -> float* {
    if (PRE_LOAD_PARAM) return l;
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
    auto conv_out = ComputeLayerConv2d<PRE_LOAD_PARAM>(
        in_data, hi, wi, h2, w2, c2, bottleneck_layer_name + std::string("_downsample_conv2d"));
    auto short_cut_out = ComputeLayerBatchNorm<PRE_LOAD_PARAM>(
        conv_out, h2, w2, c2, bottleneck_layer_name + std::string("_downsample_batchnorm"));
    Add(bn_out, short_cut_out, bn_out, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return ComputeLayerRelu<PRE_LOAD_PARAM>(bn_out, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    Add(bn_out, in_data, bn_out, h0 * w0 * c0);
    free(in_data);
    return ComputeLayerRelu<PRE_LOAD_PARAM>(bn_out, h0 * w0 * c0);
  }
}

void PreLoadParams() {
  float* img = nullptr;
  int h0, w0, c0;
  int h1, w1, c1;
  img = ComputeLayerConv2d<true>(img, 224, 224, h1, w1, c1, "conv1");
  img = ComputeLayerBatchNorm<true>(img, h1, w1, c1, "bn1");
  img = ComputeLayerRelu<true>(img, h1 * w1 * c1);
  img = ComputeLayerMaxPool<true>(img);
  // layer1
  img = ComputeBottleNeck<true>(img, 56, 56, h1, w1, c1, "layer1_bottleneck0", true);
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer1_bottleneck1", false);
  img = ComputeBottleNeck<true>(img, h0, w0, h1, w1, c1, "layer1_bottleneck2", false);
  // layer2
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer2_bottleneck0", true);
  img = ComputeBottleNeck<true>(img, h0, w0, h1, w1, c1, "layer2_bottleneck1", false);
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer2_bottleneck2", false);
  img = ComputeBottleNeck<true>(img, h0, w0, h1, w1, c1, "layer2_bottleneck3", false);
  // layer3
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer3_bottleneck0", true);
  img = ComputeBottleNeck<true>(img, h0, w0, h1, w1, c1, "layer3_bottleneck1", false);
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer3_bottleneck2", false);
  img = ComputeBottleNeck<true>(img, h0, w0, h1, w1, c1, "layer3_bottleneck3", false);
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer3_bottleneck4", false);
  img = ComputeBottleNeck<true>(img, h0, w0, h1, w1, c1, "layer3_bottleneck5", false);
  // layer4
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer4_bottleneck0", true);
  img = ComputeBottleNeck<true>(img, h0, w0, h1, w1, c1, "layer4_bottleneck1", false);
  img = ComputeBottleNeck<true>(img, h1, w1, h0, w0, c0, "layer4_bottleneck2", false);
  // avg pool
  img = ComputeLayerAvgPool<true>(img);
  // Linear
  img = ComputeLayerFC<true>(img, "fc");
  return;
}
