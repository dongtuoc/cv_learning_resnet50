#include "conv2d.h"

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
                 bool is_free_img) {
  // 计算输出图像的高度和宽度
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  // 分配输出图像的内存
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
        float acc = 0;                           // 初始化乘累加结果

        // 遍历卷积核的高度
        for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          const int hi_index = in_h_origin + kh_idx;  // 计算当前卷积核高度的索引
          // 遍历卷积核的宽度
          for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            const int wi_index = in_w_origin + kw_idx;  // 计算当前卷积核宽度的索引
            // 遍历输入通道
            for (int ci_ = 0; ci_ < ci; ci_++) {
              auto in_data = img[hi_index * wi * ci + wi_index * ci + ci_];  // 获取输入图像的数据
              auto weight_data = weight[co_idx * kernel * kernel * ci + kh_idx * kernel * ci +
                                        kw_idx * ci + ci_];  // 获取权重数据
              acc += in_data * weight_data;  // 执行卷积运算，本质就是乘累加运算
            }
          }
        }
        out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;  // 存储卷积结果
      }
    }
  }

  if (is_free_img) {
    free(img);
  }
  free(weight);
  return out;  // 返回卷积后的输出图像
}