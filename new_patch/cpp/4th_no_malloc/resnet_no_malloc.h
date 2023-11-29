#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// unlikely to predict un-commonly-used branches
#define unlikely(x) __builtin_expect(!!(x), 0)

// set DEBUG_SHOW to 1 to enable printf function
#define DEBUG_SHOW (0)

inline float avx2_sum(__m256 in_vec) {
  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));
  in_vec = _mm256_hadd_ps(in_vec, in_vec);
  float sum0 = _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));
  return sum0;
}

template <bool First, bool PRE_LOAD_PARAM>
static void MyConv2d(void* img_in,
                     void* img_out,
                     float* weight,
                     int hi,
                     int wi,
                     int& ho,
                     int& wo,
                     int ci,
                     int co,
                     int kernel,
                     int stride,
                     int pad) {
#if DEBUG_SHOW
  printf("conv in: (%d, %d, %d)\n", hi, wi, ci);
#endif
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  if (PRE_LOAD_PARAM) return;

  float* img = (float*)img_in;
  float* out = (float*)img_out;

  for (int co_idx = 0; co_idx < co; co_idx++) {
    int co_idx_for_cal = co_idx * kernel * kernel * ci;
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
                acc += avx2_sum(in_vec);
              }
            }
          }
          out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;
        }
      }
    }
  }

#if DEBUG_SHOW
  printf("conv out: (%d, %d, %d)\n", ho, wo, co);
#endif
}

template <bool PRE_LOAD_PARAM>
static void MyFC(void* img_in, void* img_out, float* weight, float* bias) {
#if DEBUG_SHOW
  printf("fc in: (1000, 2048)\n");
  printf("fc out: (1000)\n");
#endif
  if (PRE_LOAD_PARAM) return;
  float* img = (float*)img_in;
  float* out = (float*)img_out;
  for (int i = 0; i < 1000; i++) {
    float sum_x = float(0);
#if 0
    for (int j = 0; j < 2048; j++) {
      auto l = img[j];
      auto r = weight[i * 2048 + j];
      sum_x += l * r;
    }
#else
    const int vec_size = 8;
    __m256 l_vec, weight_vec;
    for (int j = 0; j < 2048; j += vec_size) {
      l_vec = _mm256_loadu_ps(&img[j]);
      weight_vec = _mm256_loadu_ps(&weight[i * 2048 + j]);
      l_vec = _mm256_mul_ps(l_vec, weight_vec);
      // Add the elements of the accumulator vector and store the result
      sum_x += avx2_sum(l_vec);
    }
#endif
    out[i] = sum_x + bias[i];
  }
  return;
}

template <bool PRE_LOAD_PARAM>
static void MyMaxPool(void* img_in, void* img_out) {
  if (PRE_LOAD_PARAM) return;
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
  float* img = (float*)img_in;
  float* out = (float*)img_out;

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
#if DEBUG_SHOW
  printf("maxpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
}

template <bool PRE_LOAD_PARAM>
static void MyAvgPool(void* img_in, void* img_out) {
  if (PRE_LOAD_PARAM) return;
  const auto hi = 7;
  const auto wi = 7;
  const auto channel = 2048;
  const auto pad = 0;
  const auto stride = 1;
  const auto kernel = 7;
  const auto ho = (hi + 2 * pad - kernel) / stride + 1;
  const auto wo = (wi + 2 * pad - kernel) / stride + 1;
  float* img = (float*)img_in;
  float* out = (float*)img_out;
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
#if DEBUG_SHOW
  printf("avgpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
}

template <bool PRE_LOAD_PARAM>
static void MyBatchNorm(void* img_in,
                        void* img_out,
                        float* mean,
                        float* var,
                        float* gamma,
                        float* bias,
                        int h,
                        int w,
                        int c) {
#if DEBUG_SHOW
  printf("bn in : (%d, %d, %d)\n", h, w, c);
#endif
  if (PRE_LOAD_PARAM) return;
  float* img = (float*)img_in;
  float* out = (float*)img_out;
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
#if DEBUG_SHOW
  printf("bn out: (%d, %d, %d)\n", h, w, c);
#endif
}

// Relu Do Inplace Computation
template <bool PRE_LOAD_PARAM>
static void ComputeLayerRelu(void* img_in, int len) {
#if DEBUG_SHOW
  printf("-- compute relu with %d\n", len);
#endif
  if (PRE_LOAD_PARAM) {
    return;
  } else {
    float* img = (float*)img_in;
    for (int i = 0; i < len; i++) {
      img[i] = img[i] > 0 ? img[i] : 0;
    }
  }
}

// optimize by pre-load params of networks
#define MAX_MEM_NUM (1024)
static void* __global_weight[MAX_MEM_NUM] = {nullptr};
int put_cnt = 0;
int out_cnt = 0;

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
  __global_weight[put_cnt++] = data;
  return (void*)data;
}

template <bool PRE_LOAD_PARAM>
static float* LoadCon2dWeight(const std::string& name, int len) {
  if (PRE_LOAD_PARAM) {
    auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
    return (float*)LoadData<float>(file_name, len, true);
  } else {
    return (float*)__global_weight[out_cnt++];
  }
}

template <bool PRE_LOAD_PARAM>
static int* LoadCon2dParam(const std::string& name, int len) {
  if (PRE_LOAD_PARAM) {
    auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
    return (int*)LoadData<int>(file_name, len, false);
  } else {
    return (int*)__global_weight[out_cnt++];
  }
}

template <bool PRE_LOAD_PARAM>
static void ComputeLayerConv2d(void* img_in,
                               void* img_out,
                               int hi,
                               int wi,
                               int& ho,
                               int& wo,
                               int& co,
                               const std::string& layer_name) {
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
    return MyConv2d<true, PRE_LOAD_PARAM>(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel,
                                          stride, pad);
  } else {
    return MyConv2d<false, PRE_LOAD_PARAM>(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel,
                                           stride, pad);
  }
}

template <bool PRE_LOAD_PARAM>
static void ComputeLayerFC(void* img_in, void* img_out, const std::string& layer_name) {
#if DEBUG_SHOW
  std::cout << "-- compute " << layer_name << std::endl;
#endif
  if (PRE_LOAD_PARAM) {
    auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
    auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
    LoadData<float>(weight_file_name, 1000 * 2048, true);
    LoadData<float>(bias_file_name, 1000, true);
    return;
  } else {
    auto weight = (float*)__global_weight[out_cnt++];
    auto bias = (float*)__global_weight[out_cnt++];
    return MyFC<PRE_LOAD_PARAM>(img_in, img_out, weight, bias);
  }
}

template <bool PRE_LOAD_PARAM>
static void ComputeLayerBatchNorm(
    void* in_data, void* out_data, int h, int w, int c, const std::string& layer_name) {
#if DEBUG_SHOW
  std::cout << "-- compute " << layer_name << std::endl;
#endif
  if (PRE_LOAD_PARAM) {
    auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
    auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
    auto mean_file_name =
        "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
    auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
    LoadData<float>(weight_file_name, c, true);
    LoadData<float>(bias_file_name, c, true);
    LoadData<float>(mean_file_name, c, true);
    LoadData<float>(var_file_name, c, true);
    return;
  } else {
    auto gamma = (float*)__global_weight[out_cnt++];
    auto bias = (float*)__global_weight[out_cnt++];
    auto mean = (float*)__global_weight[out_cnt++];
    auto var = (float*)__global_weight[out_cnt++];
    return MyBatchNorm<PRE_LOAD_PARAM>(in_data, out_data, mean, var, gamma, bias, h, w, c);
  }
}

template <bool PRE_LOAD_PARAM>
static void ComputeLayerMaxPool(void* in_data, void* out_data) {
#if DEBUG_SHOW
  std::cout << "-- compute maxpool" << std::endl;
#endif
  if (PRE_LOAD_PARAM) {
    return;
  } else {
    return MyMaxPool<PRE_LOAD_PARAM>(in_data, out_data);
  }
}

template <bool PRE_LOAD_PARAM>
static void ComputeLayerAvgPool(void* in_data, void* out_data) {
#if DEBUG_SHOW
  std::cout << "-- compute avgpool" << std::endl;
#endif
  if (PRE_LOAD_PARAM) {
    return;
  } else {
    return MyAvgPool<PRE_LOAD_PARAM>(in_data, out_data);
  }
}

template <bool PRE_LOAD_PARAM>
static void Add(float* l, float* r, float* out, int len) {
  if (PRE_LOAD_PARAM) return;
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

template <bool PRE_LOAD_PARAM>
static void ComputeBottleNeck(void* in_data,
                              void* out_data,
                              void* temp_data,
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

  ComputeLayerConv2d<PRE_LOAD_PARAM>(in_data, out_data, hi, wi, h0, w0, c0,
                                     bottleneck_layer_name + "_conv1");
  ComputeLayerBatchNorm<PRE_LOAD_PARAM>(out_data, temp_data, h0, w0, c0,
                                        bottleneck_layer_name + std::string("_bn1"));
  ComputeLayerRelu<PRE_LOAD_PARAM>(temp_data, h0 * w0 * c0);

  ComputeLayerConv2d<PRE_LOAD_PARAM>(temp_data, out_data, h0, w0, h1, w1, c1,
                                     bottleneck_layer_name + std::string("_conv2"));
  ComputeLayerBatchNorm<PRE_LOAD_PARAM>(out_data, temp_data, h1, w1, c1,
                                        bottleneck_layer_name + std::string("_bn2"));
  ComputeLayerRelu<PRE_LOAD_PARAM>(temp_data, h1 * w1 * c1);

  ComputeLayerConv2d<PRE_LOAD_PARAM>(temp_data, out_data, h1, w1, h0, w0, c0,
                                     bottleneck_layer_name + std::string("_conv3"));
  ComputeLayerBatchNorm<PRE_LOAD_PARAM>(out_data, temp_data, h0, w0, c0,
                                        bottleneck_layer_name + std::string("_bn3"));
  auto bn_out = temp_data;

  if (unlikely(down_sample)) {
    int h2, w2, c2;
    ComputeLayerConv2d<PRE_LOAD_PARAM>(in_data, out_data, hi, wi, h2, w2, c2,
                                       bottleneck_layer_name + std::string("_downsample_conv2d"));
    ComputeLayerBatchNorm<PRE_LOAD_PARAM>(
        out_data, in_data, h2, w2, c2,
        bottleneck_layer_name + std::string("_downsample_batchnorm"));
    auto short_cut_out = in_data;
    Add<PRE_LOAD_PARAM>((float*)bn_out, (float*)short_cut_out, (float*)out_data, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return ComputeLayerRelu<PRE_LOAD_PARAM>(out_data, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    Add<PRE_LOAD_PARAM>((float*)bn_out, (float*)in_data, (float*)out_data, h0 * w0 * c0);
    return ComputeLayerRelu<PRE_LOAD_PARAM>(out_data, h0 * w0 * c0);
  }
}

void PreLoadParams() {
  float* img0 = nullptr;
  float* img1 = nullptr;
  float* img2 = nullptr;
  int h0, w0, c0;
  int h1, w1, c1;
  ComputeLayerConv2d<true>(img0, img1, 224, 224, h1, w1, c1, "conv1");
  ComputeLayerBatchNorm<true>(img1, img0, h1, w1, c1, "bn1");
  ComputeLayerRelu<true>(img0, h1 * w1 * c1);
  ComputeLayerMaxPool<true>(img0, img1);
  // layer1
  ComputeBottleNeck<true>(img1, img0, img2, 56, 56, h1, w1, c1, "layer1_bottleneck0", true);
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer1_bottleneck1", false);
  ComputeBottleNeck<true>(img1, img0, img2, h0, w0, h1, w1, c1, "layer1_bottleneck2", false);
  // layer2
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck0", true);
  ComputeBottleNeck<true>(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck1", false);
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck2", false);
  ComputeBottleNeck<true>(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck3", false);
  // layer3
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck0", true);
  ComputeBottleNeck<true>(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck1", false);
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck2", false);
  ComputeBottleNeck<true>(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck3", false);
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck4", false);
  ComputeBottleNeck<true>(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck5", false);
  // layer4
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck0", true);
  ComputeBottleNeck<true>(img1, img0, img2, h0, w0, h1, w1, c1, "layer4_bottleneck1", false);
  ComputeBottleNeck<true>(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck2", false);
  // avg pool
  ComputeLayerAvgPool<true>(img1, img0);
  // Linear
  ComputeLayerFC<true>(img0, img1, "fc");
  return;
}
