#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

// 使用 AVX2 指令集加速进行向量求和
static inline float avx2_sum(__m256 in_vec) {
  // 将输入向量与其自身的高低位交换相加
  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));
  // 使用水平加法合并向量元素
  in_vec = _mm256_hadd_ps(in_vec, in_vec);
  // 再次使用水平加法并从结果向量中提取最终求和结果
  float sum0 = _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));
  return sum0;
}

// 定义预加载版本的 MyConv2d 函数，仅计算输出尺寸，不执行实际卷积运算
void MyConv2dPreLoad(void* img_in,
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
  // 计算输出高度和宽度
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  return;
}

// 定义 MyConv2d 函数，执行实际的卷积运算
void MyConv2d(void* img_in,
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
              bool First) {
  // 计算输出高度和宽度
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;

  // 类型转换输入和输出为 float 指针
  float* img = (float*)img_in;
  float* out = (float*)img_out;

  // 遍历输出特征图的每个元素
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
          // 对于第一层卷积，使用标准的卷积操作
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
          // 对于后续层，使用 AVX2 向量指令优化卷积操作
          const int vec_size = 8;
          for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
            const register int hi_index = in_h_origin + kh_idx;
            for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
              const register int wi_index = in_w_origin + kw_idx;
              // 加载输入和权重数据到向量中
              __m256 in_vec, weight_vec;  // 定义两个向量寄存器，存放向量数据
              for (int ci_ = 0; ci_ < ci; ci_ += vec_size) {
                in_vec = _mm256_loadu_ps(
                    &img[hi_index * wi * ci + wi_index * ci + ci_]);  // 加载feature map
                weight_vec = _mm256_loadu_ps(&weight[co_idx * kernel * kernel * ci +  // 加载 weight
                                                     kh_idx * kernel * ci + kw_idx * ci + ci_]);
                in_vec = _mm256_mul_ps(in_vec, weight_vec);  // feature map 和 weight 相乘
                // 将累加器向量的元素相加并存储结果
                acc += avx2_sum(in_vec);
              }
            }
          }
          out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;
        }
      }
    }
  }
}
