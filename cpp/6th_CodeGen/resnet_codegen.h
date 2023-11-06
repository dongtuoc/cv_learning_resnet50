#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define WARM_UP (true)
#define NO_WARM_UP (false)
#define INFER (NO_WARM_UP)

#define PRELOAD (true)
#define NO_PRELOAD (false)

std::ostringstream os;

void writeStringToFile(const std::string& content) {
  std::ofstream file("lib/main.cc");

  if (file.is_open()) {
    file << content;
    file.close();
    std::cout << "succ to lib/main.cc" << std::endl;
  } else {
    std::cout << "fail to open lib/main.cc" << std::endl;
  }
  exit(0);
}

void Gen(const std::string& var) { os << var; }

// sum vector by avx
inline float avx2_sum(__m256 in_vec) {
  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));
  in_vec = _mm256_hadd_ps(in_vec, in_vec);
  return _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));
}

std::vector<std::vector<std::vector<int>>> __globle_conv_input_idx;
std::vector<std::vector<std::vector<int>>> __globle_conv_weight_idx;
std::vector<std::vector<int>> __globle_conv_output_idx;
int conv_idx = 0;
int bottle_idx = 0;

void PrintVec(std::ostringstream& os, const std::vector<int> vec, const std::string& name) {
  os << "int " << name << "[" << vec.size() << "] = {";
  for (auto it : vec) {
    os << it << ", ";
  }
  os << "};\n";
}

static void MyConv2dInfer(void* img_in, void* img_out, float* weight) {
  os << "static void MyConv2dInfer_" << conv_idx
     << "(void* img_in, void* img_out, float* weight) {\n";
  float* img = (float*)img_in;
  float* out = (float*)img_out;
  os << "float* img = (float*)img_in;\n";
  os << "float* out = (float*)img_out;\n";
#if 0
  const auto& this_conv_input = __globle_conv_input_idx[conv_idx];
  const auto& this_conv_weight = __globle_conv_weight_idx[conv_idx];
  const auto& this_conv_out = __globle_conv_output_idx[conv_idx];

  const auto& this_conv_loop = this_conv_input.size();
  for (int i = 0; i < this_conv_loop; ++i) {
    os << "{\n";
    const auto& channel = this_conv_input[i];
    const auto& weight_ii = this_conv_weight[i];
    PrintVec(os, channel, "channel");
    PrintVec(os, weight_ii, "weight_ii");

    float acc = 0;
    os << "float acc = 0;\n";
    const int& channel_size = channel.size();
    if (conv_idx == 0) {
      os << "for (auto idx = 0; idx <" << channel_size << "; ++idx) {\n";
      os << "  acc += img[channel[idx]] * weight[weight_ii[idx]];\n";
      os << "}\n";
      for (auto idx = 0; idx < channel_size; ++idx) {
        acc += img[channel[idx]] * weight[weight_ii[idx]];
      }
    } else {
      __m256 in_vec, weight_vec;
      os << "__m256 in_vec, weight_vec;";
      os << "for (auto idx = 0; idx < " << channel_size << "; idx ++) {\n";
      os << "in_vec = _mm256_loadu_ps(&img[channel[idx]]);\n";
      os << "weight_vec = _mm256_loadu_ps(&weight[weight_ii[idx]]);\n";
      os << "in_vec = _mm256_mul_ps(in_vec, weight_vec);\n";
      os << "acc += avx2_sum(in_vec);\n";
      os << "}\n";

      for (auto idx = 0; idx < channel_size; ++idx) {
        in_vec = _mm256_loadu_ps(&img[channel[idx]]);
        weight_vec = _mm256_loadu_ps(&weight[weight_ii[idx]]);
        in_vec = _mm256_mul_ps(in_vec, weight_vec);
        // Add the elements of the accumulator vector and store the result
        acc += avx2_sum(in_vec);
      }
    }
    out[this_conv_out[i]] = acc;
    os << "out[" << this_conv_out[i] << "] = acc;\n";
    os << "}";
  }
#endif
  os << "}\n";
  conv_idx++;
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
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
                     int pad,
                     bool first) {
  if (PRE_LOAD_PARAM) return;

  if (WARM_UP_FOR_INFER) {
    ho = (hi + 2 * pad - kernel) / stride + 1;
    wo = (wi + 2 * pad - kernel) / stride + 1;
    float* img = (float*)img_in;
    float* out = (float*)img_out;

    std::vector<std::vector<int>> input_conv_idx;
    std::vector<std::vector<int>> weight_conv_idx;
    std::vector<int> out_conv_idx;

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
          std::vector<int> input_channel_idx;
          std::vector<int> weight_channel_idx;
          if (first) {
            for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
              const int hi_index = in_h_origin + kh_idx;
              for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
                const int wi_index = in_w_origin + kw_idx;
                for (int ci_ = 0; ci_ < 3; ci_++) {
                  auto in_data = img[hi_index * 224 * 3 + wi_index * 3 + ci_];
                  auto weight_data = weight[co_idx * 49 * 3 + kh_idx * 7 * 3 + kw_idx * 3 + ci_];
                  acc += in_data * weight_data;

                  input_channel_idx.push_back(hi_index * 224 * 3 + wi_index * 3 + ci_);
                  weight_channel_idx.push_back(co_idx * 49 * 3 + kh_idx * 7 * 3 + kw_idx * 3 + ci_);
                }
              }
            }
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

                  input_channel_idx.push_back(hi_index * wi * ci + wi_index * ci + ci_);
                  weight_channel_idx.push_back(co_idx * kernel * kernel * ci +
                                               kh_idx * kernel * ci + kw_idx * ci + ci_);
                }
              }
            }
          }
          out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;

          input_conv_idx.push_back(input_channel_idx);
          weight_conv_idx.push_back(weight_channel_idx);
          out_conv_idx.push_back(ho_idx * wo * co + wo_idx * co + co_idx);
        }
      }
    }
    __globle_conv_input_idx.push_back(input_conv_idx);
    __globle_conv_weight_idx.push_back(weight_conv_idx);
    __globle_conv_output_idx.push_back(out_conv_idx);
  }
}

std::vector<std::vector<int>> __globle_fc_input_idx;
std::vector<std::vector<int>> __globle_fc_weight_idx;
std::vector<int> __globle_fc_output_idx;

static void MyFullyConnectInfer(void* img_in, void* img_out, float* weight, float* bias) {
  os << "static void MyFullyConnectInfer(void* img_in, void* img_out, float* weight, float* bias)\n"
        "{\n";
  os << "float* img = (float*)img_in;\n";
  os << "float* out = (float*)img_out;\n";
  os << "for (int outer = 0; outer < 1000; ++outer) {\n";
  os << "  float sum_x = float(0);\n";
  os << "  const int vec_size = 8;\n";
  os << "  __m256 l_vec, weight_vec;\n";
  os << "  for (int inner = 0; inner < 2048; inner += vec_size) {\n";
  os << "    l_vec = _mm256_loadu_ps(&img[inner]);\n";
  os << "    weight_vec = _mm256_loadu_ps(&weight[outer * 2048 + inner]);\n";
  os << "    l_vec = _mm256_mul_ps(l_vec, weight_vec);\n";
  os << "    sum_x += avx2_sum(l_vec);\n";
  os << "  }\n";
  os << "  out[outer] = sum_x + bias[outer];\n";
  os << "}\n";
  os << "}\n";

  float* img = (float*)img_in;
  float* out = (float*)img_out;
  for (int outer = 0; outer < 1000; ++outer) {
    float sum_x = float(0);
    const int vec_size = 8;
    __m256 l_vec, weight_vec;
    // const auto& in_idx = __globle_fc_input_idx[outer];
    // const auto& we_idx = __globle_fc_weight_idx[outer];
    // for (int inner = 0; inner < in_idx.size(); inner++) {
    for (int inner = 0; inner < 2048; inner += vec_size) {
      l_vec = _mm256_loadu_ps(&img[inner]);
      weight_vec = _mm256_loadu_ps(&weight[outer * 2048 + inner]);
      l_vec = _mm256_mul_ps(l_vec, weight_vec);
      sum_x += avx2_sum(l_vec);
    }
    out[outer] = sum_x + bias[outer];
  }

  // writeStringToFile(os.str());
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void MyFullyConnect(void* img_in, void* img_out, float* weight, float* bias) {
  if (PRE_LOAD_PARAM) return;

  if (WARM_UP_FOR_INFER) {
    float* img = (float*)img_in;
    float* out = (float*)img_out;

    for (int i = 0; i < 1000; i++) {
      float sum_x = float(0);

      std::vector<int> in_idx;
      std::vector<int> weight_idx;
      const int vec_size = 8;
      __m256 l_vec, weight_vec;
      for (int j = 0; j < 2048; j += vec_size) {
        l_vec = _mm256_loadu_ps(&img[j]);
        weight_vec = _mm256_loadu_ps(&weight[i * 2048 + j]);
        l_vec = _mm256_mul_ps(l_vec, weight_vec);
        sum_x += avx2_sum(l_vec);

        in_idx.push_back(j);
        weight_idx.push_back(i * 2048 + j);
      }
      out[i] = sum_x + bias[i];
      __globle_fc_input_idx.push_back(in_idx);
      __globle_fc_weight_idx.push_back(weight_idx);
      __globle_fc_output_idx.push_back(i);
    }
  }
}

std::vector<std::vector<int>> __globle_maxpool_input_idx;
std::vector<int> __globle_maxpool_output_idx;

static void MyMaxPoolInfer(void* img_in, void* img_out) {
  os << "static void MyMaxPoolInfer(void* img_in, void* img_out) {\n";
  os << "  float* img = (float*)img_in;\n";
  os << "  float* out = (float*)img_out;\n";

  float* img = (float*)img_in;
  float* out = (float*)img_out;

#if 0
  for (int i = 0; i < __globle_maxpool_output_idx.size(); i++) {
    os << "{\n";
    const auto& this_pool = __globle_maxpool_input_idx[i];
    PrintVec(os, this_pool, "this_pool");
    float max_x = float(0);
    os << "float max_x = float(0);\n";
    os << "for (auto j = 0; j < this_pool.size(); j++) {\n";
    os << "  auto in_data = img[this_pool[j]]\n";
    os << "  max_x = std::max(in_data, max_x);\n";
    os << "}\n";
    os << "out[" << __globle_maxpool_output_idx[i] << "] = max_x;\n";
    os << "}\n";
    for (auto j = 0; j < this_pool.size(); j++) {
      auto in_data = img[this_pool[j]];
      max_x = std::max(in_data, max_x);
    }
    out[__globle_maxpool_output_idx[i]] = max_x;
  }
#endif
  os << "}\n";
  // writeStringToFile(os.str());
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void MyMaxPool(void* img_in, void* img_out) {
  if (PRE_LOAD_PARAM) return;
  if (WARM_UP_FOR_INFER) {
    const auto hi = 112;
    const auto wi = 112;
    const auto channel = 64;
    const auto pad = 1;
    const auto stride = 2;
    const auto kernel = 3;
    const auto ho = (hi + 2 * pad - kernel) / stride + 1;
    const auto wo = (wi + 2 * pad - kernel) / stride + 1;
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

          std::vector<int> index;
          for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
            auto hi_index = in_h_origin + kh_idx;
            for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
              auto wi_index = in_w_origin + kw_idx;
              auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
              max_x = std::max(in_data, max_x);

              index.push_back(hi_index * wi * channel + wi_index * channel + c_);
            }
          }
          out[ho_idx * wo * channel + wo_idx * channel + c_] = max_x;

          __globle_maxpool_input_idx.push_back(index);
          __globle_maxpool_output_idx.push_back(ho_idx * wo * channel + wo_idx * channel + c_);
        }
      }
    }
  }
}

std::vector<std::vector<int>> __globle_avgpool_input_idx;
std::vector<int> __globle_avgpool_output_idx;

static void MyAvgPoolInfer(void* img_in, void* img_out) {
  os << "static void MyAvgPoolInfer(void* img_in, void* img_out) {\n";
  os << "  float* img = (float*)img_in;\n";
  os << "  float* out = (float*)img_out;\n";

  float* img = (float*)img_in;
  float* out = (float*)img_out;

  // for (int i = 0; i < __globle_avgpool_output_idx.size(); i++) {
  //  os << "{\n";
  //   const auto& this_pool = __globle_avgpool_input_idx[i];
  //   PrintVec(os, this_pool, "this_pool");
  //   float sum = float(0);
  //   os << "float sum = float(0);\n";
  //   os << "for (auto j = 0; j < this_pool.size(); j++) {\n";
  //   os << "  auto in_data = img[this_pool[j]];\n";
  //   os << "  sum += in_data;\n";
  //   os << "}\n";
  //   os << "out[" << __globle_avgpool_output_idx[i] << "] = "
  //      << "sum / " << this_pool.size() << ";\n";
  //   os << "}";
  //   for (auto j = 0; j < this_pool.size(); j++) {
  //     auto in_data = img[this_pool[j]];
  //     sum += in_data;
  //   }
  //   out[__globle_avgpool_output_idx[i]] = sum / this_pool.size();
  // }
  os << "}";
  // writeStringToFile(os.str());
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void MyAvgPool(void* img_in, void* img_out) {
  if (PRE_LOAD_PARAM) return;
  if (WARM_UP_FOR_INFER) {
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

          std::vector<int> index;

          for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
            auto hi_index = in_h_origin + kh_idx;
            for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
              auto wi_index = in_w_origin + kw_idx;
              auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
              sum += in_data;

              index.push_back(hi_index * wi * channel + wi_index * channel + c_);
            }
          }
          out[ho_idx * wo * channel + wo_idx * channel + c_] = sum / k_size;

          __globle_avgpool_output_idx.push_back(ho_idx * wo * channel + wo_idx * channel + c_);
          __globle_avgpool_input_idx.push_back(index);
        }
      }
    }
  }
}

std::vector<std::vector<std::vector<int>>> __globle_bn_in_out_idx;
std::vector<std::vector<int>> __globle_bn_mean_v_gm_bias_idx;
int bn_cnt = 0;

static void MyBatchNormInfer(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
  os << "static void MyBatchNormInfer_" << bn_cnt << "(\n";
  os << "    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {\n";
  os << "  float* img = (float*)img_in;\n";
  os << "  float* out = (float*)img_out;\n";

  const auto& this_bn_in_out = __globle_bn_in_out_idx[bn_cnt];
  const auto& this_bn_param = __globle_bn_mean_v_gm_bias_idx[bn_cnt];

#if 0
  for (int i = 0; i < this_bn_in_out.size(); i++) {
    os << "for (int i = 0; i <" << this_bn_in_out.size() << "; i++) {";
    const int out_index = this_bn_param[i];
    auto m = mean[i];
    auto v = var[i];
    auto gm = gamma[i];
    auto bi = bias[i];
    os << "auto m = mean[i];";
    os << "auto v = var[i];";
    os << "auto gm = gamma[i];";
    os << "auto bi = bias[i];";
    const auto& channel = this_bn_in_out[i];
    const float div = sqrt(v + 1e-5);
    os << "const float div = sqrt(v + 1e-5);";
    for (auto idx = 0; idx < channel.size(); idx++) {
      os << "for (auto idx = 0; idx < " << channel.size() << "; idx++) {";
      auto data_ = (img[channel[idx]] - m) / div;
      os << "auto data_ = (img[" << channel[idx] << "] - m) / div;";
      data_ = data_ * gm + bi;
      os << "data_ = data_ * gm + bi;";
      out[channel[idx]] = data_;
      os << "out[" << channel[idx] << "] = data_;";
    }
    os << "}";
  }
#endif
  os << "}\n";
  bn_cnt++;
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void MyBatchNorm(void* img_in,
                        void* img_out,
                        float* mean,
                        float* var,
                        float* gamma,
                        float* bias,
                        int h,
                        int w,
                        int c) {
  if (PRE_LOAD_PARAM) return;
  if (WARM_UP_FOR_INFER) {
    float* img = (float*)img_in;
    float* out = (float*)img_out;

    std::vector<std::vector<int>> in_out;
    std::vector<int> param;
    for (auto c_ = 0; c_ < c; c_++) {
      auto m = mean[c_];
      auto v = var[c_];
      auto gm = gamma[c_];
      auto bi = bias[c_];

      std::vector<int> index_in;
      for (auto hw = 0; hw < h * w; hw++) {
        auto data = img[hw * c + c_];
        auto data_ = (data - m) / sqrt(v + 1e-5);
        data_ = data_ * gm + bi;
        out[hw * c + c_] = data_;

        index_in.push_back(hw * c + c_);
      }
      in_out.push_back(index_in);
      param.push_back(c_);
    }

    __globle_bn_in_out_idx.push_back(in_out);
    __globle_bn_mean_v_gm_bias_idx.push_back(param);
  }
}

std::vector<int> __globle_relu_in_out_len;
int relu_cnt = 0;

// Relu Do Inplace Computation
static void InferLayerRelu(void* img_in) {
  os << "static void InferLayerRelu_" << relu_cnt << "(void* img_in) {\n";
  os << "  const auto& len = " << __globle_relu_in_out_len[relu_cnt] << ";\n";
  os << "  float* img = (float*)img_in;\n";
  os << "  for (int i = 0; i < len; ++i) {\n";
  os << "    img[i] = img[i] > 0 ? img[i] : 0;\n";
  os << "  }\n";
  os << "}\n";

  const auto& len = __globle_relu_in_out_len[relu_cnt];
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
  relu_cnt++;
}

// Relu Do Inplace Computation
template <bool PRE_LOAD_PARAM>
static void ComputeLayerRelu(void* img_in, int len) {
  if (PRE_LOAD_PARAM) {
    return;
  } else {
    float* img = (float*)img_in;

    for (int i = 0; i < len; i++) {
      img[i] = img[i] > 0 ? img[i] : 0;
    }

    __globle_relu_in_out_len.push_back(len);
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

static void InferLayerConv2d(void* img_in, void* img_out) {
  // not used
  // auto param = (float*)__global_weight[out_cnt++];
  out_cnt++;
  auto weight = (float*)__global_weight[out_cnt++];
  MyConv2dInfer(img_in, img_out, weight);

  os << "static void InferLayerConv2d_" << conv_idx - 1 << "(void* img_in, void* img_out) {\n";
  os << "  MyConv2dInfer_" << conv_idx - 1 << "(img_in, img_out, (float *)" << weight << ");\n";
  os << "}\n";
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void ComputeLayerConv2d(void* img_in,
                               void* img_out,
                               int hi,
                               int wi,
                               int& ho,
                               int& wo,
                               int& co,
                               const std::string& layer_name) {
  auto param = LoadCon2dParam<PRE_LOAD_PARAM>(layer_name, 5);
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeight<PRE_LOAD_PARAM>(layer_name, co * kernel * kernel * ci);
  return MyConv2d<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(img_in, img_out, weight, hi, wi, ho, wo, ci,
                                                     co, kernel, stride, pad, hi == 224);
}

static void InferLayerFC(void* img_in, void* img_out) {
  auto weight = (float*)__global_weight[out_cnt++];
  auto bias = (float*)__global_weight[out_cnt++];
  MyFullyConnectInfer(img_in, img_out, weight, bias);

  os << "static void InferLayerFC(void* img_in, void* img_out) {\n";
  os << "  MyFullyConnectInfer(img_in, img_out, (float *)" << weight << ", (float *)" << bias
     << ");\n";
  os << "}\n";
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void ComputeLayerFC(void* img_in, void* img_out, const std::string& layer_name) {
  if (PRE_LOAD_PARAM) {
    auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
    auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
    LoadData<float>(weight_file_name, 1000 * 2048, true);
    LoadData<float>(bias_file_name, 1000, true);
    return;
  } else {
    auto weight = (float*)__global_weight[out_cnt++];
    auto bias = (float*)__global_weight[out_cnt++];
    return MyFullyConnect<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(img_in, img_out, weight, bias);
  }
}

static void InferLayerBatchNorm(void* in_data, void* out_data) {
  auto gamma = (float*)__global_weight[out_cnt++];
  auto bias = (float*)__global_weight[out_cnt++];
  auto mean = (float*)__global_weight[out_cnt++];
  auto var = (float*)__global_weight[out_cnt++];
  MyBatchNormInfer(in_data, out_data, mean, var, gamma, bias);
  os << "static void InferLayerBatchNorm_" << bn_cnt - 1 << "(void* in_data, void* out_data) {\n";
  os << "  MyBatchNormInfer_" << bn_cnt - 1 << "(in_data, out_data, (float *)" << mean
     << ", (float *)" << var << ", (float *)" << gamma << ", (float *)" << bias << ");\n";
  os << "}\n";
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void ComputeLayerBatchNorm(
    void* in_data, void* out_data, int h, int w, int c, const std::string& layer_name) {
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
    return MyBatchNorm<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(in_data, out_data, mean, var, gamma, bias,
                                                          h, w, c);
  }
}

static void InferLayerMaxPool(void* in_data, void* out_data) {
  MyMaxPoolInfer(in_data, out_data);
  os << "static void InferLayerMaxPool(void* in_data, void* out_data) {\n";
  os << "  MyMaxPoolInfer(in_data, out_data);\n";
  os << "}\n";
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void ComputeLayerMaxPool(void* in_data, void* out_data) {
  if (PRE_LOAD_PARAM) {
    return;
  } else {
    return MyMaxPool<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(in_data, out_data);
  }
}

static void InferLayerAvgPool(void* in_data, void* out_data) {
  MyAvgPoolInfer(in_data, out_data);
  os << "static void InferLayerAvgPool(void* in_data, void* out_data) {\n";
  os << "  MyAvgPoolInfer(in_data, out_data);\n";
  os << "};\n";
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
static void ComputeLayerAvgPool(void* in_data, void* out_data) {
  if (PRE_LOAD_PARAM) {
    return;
  } else {
    return MyAvgPool<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(in_data, out_data);
  }
}

std::vector<int> __globle_add_in_out_len;
int add_cnt = 0;

static void InferLayerAdd(float* l, float* r, float* out) {
  os << "static void InferLayerAdd_" << add_cnt << "(float* l, float* r, float* out) {\n";
  os << "  const auto& len = " << __globle_add_in_out_len[add_cnt] << ";\n";
  os << "  const int vec_size = 8;\n";
  os << "  __m256 l_vec, r_vec, res_vec;\n";
  os << "  for (int i = 0; i < len; i += vec_size) {\n";
  os << "    l_vec = _mm256_loadu_ps(l + i);\n";
  os << "    r_vec = _mm256_loadu_ps(r + i);\n";
  os << "    res_vec = _mm256_add_ps(l_vec, r_vec);\n";
  os << "    _mm256_storeu_ps(out + i, res_vec);\n";
  os << "  }\n";
  os << "}\n";

  const auto& len = __globle_add_in_out_len[add_cnt];
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
  add_cnt++;
  // writeStringToFile(os.str());
  return;
}

template <bool PRE_LOAD_PARAM>
static void Add(float* l, float* r, float* out, int len) {
  if (PRE_LOAD_PARAM) return;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
  __globle_add_in_out_len.push_back(len);
  return;
}

static void InferBottleNeck(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  auto this_conv_idx = conv_idx;
  auto this_bn_idx = bn_cnt;
  auto this_relu_idx = relu_cnt;
  auto this_add_idx = add_cnt;

  InferLayerConv2d(in_data, out_data);
  InferLayerBatchNorm(out_data, temp_data);
  InferLayerRelu(temp_data);

  InferLayerConv2d(temp_data, out_data);
  InferLayerBatchNorm(out_data, temp_data);
  InferLayerRelu(temp_data);

  InferLayerConv2d(temp_data, out_data);
  InferLayerBatchNorm(out_data, temp_data);
  auto bn_out = temp_data;

  if (down_sample) {
    InferLayerConv2d(in_data, out_data);
    InferLayerBatchNorm(out_data, in_data);
    auto short_cut_out = in_data;
    InferLayerAdd((float*)bn_out, (float*)short_cut_out, (float*)out_data);
    InferLayerRelu(out_data);
  } else {
    InferLayerAdd((float*)bn_out, (float*)in_data, (float*)out_data);
    InferLayerRelu(out_data);
  }

  os << "static void InferBottleNeck_" << bottle_idx
     << "(void* in_data, void* out_data, void* temp_data, bool down_sample) {\n";
  os << "  InferLayerConv2d_" << this_conv_idx++ << "(in_data, out_data);\n";
  os << "  InferLayerBatchNorm_" << this_bn_idx++ << "(out_data, temp_data);\n";
  os << "  InferLayerRelu_" << this_relu_idx++ << "(temp_data);\n";

  os << "  InferLayerConv2d_" << this_conv_idx++ << "(temp_data, out_data);\n";
  os << "  InferLayerBatchNorm_" << this_bn_idx++ << "(out_data, temp_data);\n";
  os << "  InferLayerRelu_" << this_relu_idx++ << "(temp_data);\n";

  os << "  InferLayerConv2d_" << this_conv_idx++ << "(temp_data, out_data);\n";
  os << "  InferLayerBatchNorm_" << this_bn_idx++ << "(out_data, temp_data);\n";
  os << "  auto bn_out = temp_data;\n";
  if (down_sample) {
    os << "  InferLayerConv2d_" << this_conv_idx++ << "(in_data, out_data);\n";
    os << "  InferLayerBatchNorm_" << this_bn_idx++ << "(out_data, in_data);\n";
    os << "  auto short_cut_out = in_data;\n";
    os << "  InferLayerAdd_" << this_add_idx++
       << "((float*)bn_out, (float*)short_cut_out, (float*)out_data);\n";
    os << "  InferLayerRelu_" << this_relu_idx++ << "(temp_data);\n";
  } else {
    os << "  InferLayerAdd_" << this_add_idx++
       << "((float*)bn_out, (float*)in_data, (float*)out_data);\n";
    os << "  InferLayerRelu_" << this_relu_idx++ << "(out_data);\n";
  }
  os << "}\n";

  bottle_idx++;
}

template <bool PRE_LOAD_PARAM, bool WARM_UP_FOR_INFER>
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
  int h0, w0, c0;
  int h1, w1, c1;

  ComputeLayerConv2d<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(in_data, out_data, hi, wi, h0, w0, c0,
                                                        bottleneck_layer_name + "_conv1");
  ComputeLayerBatchNorm<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(
      out_data, temp_data, h0, w0, c0, bottleneck_layer_name + std::string("_bn1"));
  ComputeLayerRelu<PRE_LOAD_PARAM>(temp_data, h0 * w0 * c0);

  ComputeLayerConv2d<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(
      temp_data, out_data, h0, w0, h1, w1, c1, bottleneck_layer_name + std::string("_conv2"));
  ComputeLayerBatchNorm<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(
      out_data, temp_data, h1, w1, c1, bottleneck_layer_name + std::string("_bn2"));
  ComputeLayerRelu<PRE_LOAD_PARAM>(temp_data, h1 * w1 * c1);

  ComputeLayerConv2d<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(
      temp_data, out_data, h1, w1, h0, w0, c0, bottleneck_layer_name + std::string("_conv3"));
  ComputeLayerBatchNorm<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(
      out_data, temp_data, h0, w0, c0, bottleneck_layer_name + std::string("_bn3"));
  auto bn_out = temp_data;

  if (down_sample) {
    int h2, w2, c2;
    ComputeLayerConv2d<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(
        in_data, out_data, hi, wi, h2, w2, c2,
        bottleneck_layer_name + std::string("_downsample_conv2d"));
    ComputeLayerBatchNorm<PRE_LOAD_PARAM, WARM_UP_FOR_INFER>(
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
  ComputeLayerConv2d<PRELOAD, NO_WARM_UP>(img0, img1, 224, 224, h1, w1, c1, "conv1");
  ComputeLayerBatchNorm<PRELOAD, NO_WARM_UP>(img1, img0, h1, w1, c1, "bn1");
  ComputeLayerRelu<PRELOAD>(img0, h1 * w1 * c1);
  ComputeLayerMaxPool<PRELOAD, NO_WARM_UP>(img0, img1);
  // layer1
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, 56, 56, h1, w1, c1, "layer1_bottleneck0",
                                         true);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer1_bottleneck1",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, h0, w0, h1, w1, c1, "layer1_bottleneck2",
                                         false);
  // layer2
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck0",
                                         true);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck1",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck2",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck3",
                                         false);
  // layer3
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck0",
                                         true);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck1",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck2",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck3",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck4",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck5",
                                         false);
  // layer4
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck0",
                                         true);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img1, img0, img2, h0, w0, h1, w1, c1, "layer4_bottleneck1",
                                         false);
  ComputeBottleNeck<PRELOAD, NO_WARM_UP>(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck2",
                                         false);
  // avg pool
  ComputeLayerAvgPool<PRELOAD, NO_WARM_UP>(img1, img0);
  // Linear
  ComputeLayerFC<PRELOAD, NO_WARM_UP>(img0, img1, "fc");
  return;
}

void WarmUp(void* mem_main0, void* mem_main1, void* mem_temp) {
  std::cout << "\033[0;32mBegin Warm Up \033[0m" << std::endl;
  int h0, w0, c0;
  int h1, w1, c1;
  ComputeLayerConv2d<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, 224, 224, h1, w1, c1, "conv1");
  ComputeLayerBatchNorm<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, h1, w1, c1, "bn1");
  ComputeLayerRelu<NO_PRELOAD>(mem_main0, h1 * w1 * c1);
  ComputeLayerMaxPool<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1);
  // layer1
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, 56, 56, h1, w1, c1,
                                         "layer1_bottleneck0", true);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer1_bottleneck1", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1,
                                         "layer1_bottleneck2", false);
  // layer2
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer2_bottleneck0", true);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1,
                                         "layer2_bottleneck1", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer2_bottleneck2", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1,
                                         "layer2_bottleneck3", false);
  // layer3
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer3_bottleneck0", true);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1,
                                         "layer3_bottleneck1", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer3_bottleneck2", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1,
                                         "layer3_bottleneck3", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer3_bottleneck4", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1,
                                         "layer3_bottleneck5", false);
  // layer4
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer4_bottleneck0", true);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1,
                                         "layer4_bottleneck1", false);
  ComputeBottleNeck<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0,
                                         "layer4_bottleneck2", false);
  // avg pool
  ComputeLayerAvgPool<NO_PRELOAD, WARM_UP>(mem_main1, mem_main0);
  // Linear
  ComputeLayerFC<NO_PRELOAD, WARM_UP>(mem_main0, mem_main1, "fc");
  std::cout << "\033[0;32mWarm Up Done \033[0m" << std::endl;
}

void Infer(void* mem_main0, void* mem_main1, void* mem_temp) {
  os << "#include <immintrin.h>\n";
  os << "#include <cmath>\n";
  os << "#include <cstdint>\n";
  os << "#include <iostream>\n";
  os << "#include <opencv2/opencv.hpp>\n";
  os << "#include <string>\n";
  os << "#include <vector>\n";
  os << "inline float avx2_sum(__m256 in_vec) {\n";
  os << "  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));\n";
  os << "  in_vec = _mm256_hadd_ps(in_vec, in_vec);\n";
  os << "  return _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));\n";
  os << "}\n";

  out_cnt = 0;
  conv_idx = 0;
  bn_cnt = 0;
  relu_cnt = 0;
  add_cnt = 0;
  bottle_idx = 0;

  InferLayerConv2d(mem_main0, mem_main1);
  InferLayerBatchNorm(mem_main1, mem_main0);
  InferLayerRelu(mem_main0);
  InferLayerMaxPool(mem_main0, mem_main1);
  // layer1
  InferBottleNeck(mem_main1, mem_main0, mem_temp, true);
  InferBottleNeck(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck(mem_main1, mem_main0, mem_temp, false);
  // layer2
  InferBottleNeck(mem_main0, mem_main1, mem_temp, true);
  InferBottleNeck(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck(mem_main1, mem_main0, mem_temp, false);
  // layer3
  InferBottleNeck(mem_main0, mem_main1, mem_temp, true);
  InferBottleNeck(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck(mem_main1, mem_main0, mem_temp, false);
  // layer4
  InferBottleNeck(mem_main0, mem_main1, mem_temp, true);
  InferBottleNeck(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck(mem_main0, mem_main1, mem_temp, false);
  // avg pool
  InferLayerAvgPool(mem_main1, mem_main0);
  // Linear
  InferLayerFC(mem_main0, mem_main1);

  os << "void InferCodeGen(void* mem_main0, void* mem_main1, void* mem_temp) {\n";
  os << "InferLayerConv2d_0(mem_main0, mem_main1);\n";
  os << "InferLayerBatchNorm_0(mem_main1, mem_main0);\n";
  os << "InferLayerRelu_0(mem_main0);\n";
  os << "InferLayerMaxPool(mem_main0, mem_main1);\n";
  os << "// layer1\n";
  os << "InferBottleNeck_0(mem_main1, mem_main0, mem_temp, true);\n";
  os << "InferBottleNeck_1(mem_main0, mem_main1, mem_temp, false);\n";
  os << "InferBottleNeck_2(mem_main1, mem_main0, mem_temp, false);\n";
  os << "// layer2\n";
  os << "InferBottleNeck_3(mem_main0, mem_main1, mem_temp, true);\n";
  os << "InferBottleNeck_4(mem_main1, mem_main0, mem_temp, false);\n";
  os << "InferBottleNeck_5(mem_main0, mem_main1, mem_temp, false);\n";
  os << "InferBottleNeck_6(mem_main1, mem_main0, mem_temp, false);\n";
  os << "// layer3\n";
  os << "InferBottleNeck_7(mem_main0, mem_main1, mem_temp, true);\n";
  os << "InferBottleNeck_8(mem_main1, mem_main0, mem_temp, false);\n";
  os << "InferBottleNeck_9(mem_main0, mem_main1, mem_temp, false);\n";
  os << "InferBottleNeck_10(mem_main1, mem_main0, mem_temp, false);\n";
  os << "InferBottleNeck_11(mem_main0, mem_main1, mem_temp, false);\n";
  os << "InferBottleNeck_12(mem_main1, mem_main0, mem_temp, false);\n";
  os << "// layer4\n";
  os << "InferBottleNeck_13(mem_main0, mem_main1, mem_temp, true);\n";
  os << "InferBottleNeck_14(mem_main1, mem_main0, mem_temp, false);\n";
  os << "InferBottleNeck_15(mem_main0, mem_main1, mem_temp, false);\n";
  os << "// avg pool\n";
  os << "InferLayerAvgPool(mem_main1, mem_main0);\n";
  os << "// Linear\n";
  os << "InferLayerFC(mem_main0, mem_main1);\n";
  os << "}\n";
}
