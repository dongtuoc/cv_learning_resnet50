#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 定义一个函数，使用 AVX2 指令集对 __m256 类型的向量进行求和操作
inline float avx2_sum(__m256 in_vec) {
  // 将向量的两部分相加
  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));
  // 对相加结果的每两个元素进行水平相加
  in_vec = _mm256_hadd_ps(in_vec, in_vec);
  // 最后，对所有元素进行水平相加，并将结果转换为标量
  float sum0 = _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));
  return sum0;
}

// 定义预加载版本的全连接层函数，此版本不执行任何操作
void MyFCPreLoad(void* img_in, void* img_out, float* weight, float* bias) { return; }

// 定义全连接层函数，执行实际的计算操作
void MyFC(void* img_in, void* img_out, float* weight, float* bias) {
  // 类型转换输入和输出为 float 指针
  float* img = (float*)img_in;
  float* out = (float*)img_out;

  // 遍历输出节点
  for (int i = 0; i < 1000; i++) {
    float sum_x = 0.0;       // 初始化累加器为0
    const int vec_size = 8;  // 设置向量大小

    // 声明 AVX2 向量
    __m256 l_vec, weight_vec;

    // 以向量化的方式遍历输入特征
    for (int j = 0; j < 2048; j += vec_size) {
      // 加载输入和权重数据到向量中
      l_vec = _mm256_loadu_ps(&img[j]);
      weight_vec = _mm256_loadu_ps(&weight[i * 2048 + j]);
      // 将输入和权重向量相乘
      l_vec = _mm256_mul_ps(l_vec, weight_vec);
      // 使用 AVX2 指令求和累加器向量，并添加到 sum_x
      sum_x += avx2_sum(l_vec);
    }
    // 将累加器的结果加上偏置并存储在输出中
    out[i] = sum_x + bias[i];
  }
  return;
}
