#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 这段代码定义了一个自定义的二维卷积（convolution）函数
// my_conv2d，该函数执行卷积操作，通常用于图像处理和神经网络中 my_conv2d
// 函数接收一张图像（img）和卷积核权重（weight），以及各种与卷积相关的参数，如输入输出尺寸、通道数、卷积核大小、步长和填充。
// 函数内部通过嵌套循环对图像进行卷积操作，并将结果存储在新分配的内存中。

// 该函数使能了利用 avx2 向量指令集的优化。
float* my_conv2d(float* img,
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
                 bool is_first,
                 bool is_free_img = true) {
  // 计算输出图像的高度和宽度
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  float* out = (float*)malloc(ho * wo * co * sizeof(float));
  // 遍历输出通道
  for (int co_idx = 0; co_idx < co; co_idx++) {
    // 遍历输出图像的高度
    for (int ho_idx = 0; ho_idx < ho; ho_idx++) {
      const int in_h_origin = ho_idx * stride - pad;  // 计算输入图像的起始高度位置
      // 遍历输出图像的宽度
      for (int wo_idx = 0; wo_idx < wo; wo_idx++) {
        const int in_w_origin = wo_idx * stride - pad;  // 计算输入图像的起始宽度位置
        const int filter_h_start = std::max(0, -in_h_origin);  // 计算卷积核在高度方向的起始位置
        const int filter_w_start = std::max(0, -in_w_origin);  // 计算卷积核在宽度方向的起始位置
        const int filter_h_end =
            std::min(kernel, hi - in_h_origin);  // 计算卷积核在高度方向的结束位置
        const int filter_w_end =
            std::min(kernel, wi - in_w_origin);  // 计算卷积核在宽度方向的结束位置
        register float acc = 0;                  // 初始化乘累加结果
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
          // 利用 avx2 向量指令集完成卷积的乘累加操作
          const int vec_size = 8;  // 一个256bit的向量寄存器可以存放 8 个 float 数据
          for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
            const register int hi_index = in_h_origin + kh_idx;
            for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
              const register int wi_index = in_w_origin + kw_idx;
              __m256 in_vec,
                  weight_vec;  // 这是两个向量寄存器，长度为 256bit，可以存放 8 个float数据
              for (int ci_ = 0; ci_ < ci; ci_ += vec_size) {
                // 将输入和权值load到向量寄存器中
                in_vec = _mm256_loadu_ps(&img[hi_index * wi * ci + wi_index * ci + ci_]);
                weight_vec = _mm256_loadu_ps(&weight[co_idx * kernel * kernel * ci +
                                                     kh_idx * kernel * ci + kw_idx * ci + ci_]);
                // 向量乘
                in_vec = _mm256_mul_ps(in_vec, weight_vec);
                // 对乘的结果进行累加操作
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
  return out;
}