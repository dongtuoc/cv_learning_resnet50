#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// BN 层在 preLoad 中，无需做任何事情，直接返回相关的指针即可。
float* MyBatchNormPreLoad(
    float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c) {
  return img;
}

// 这段代码实现了一个批归一化（Batch Normalization）操作。
// 批归一化是深度学习中常用的一种技术，用于标准化神经网络层的输入，以加速训练过程并提高模型稳定性。
// 这个函数接收一张图像（img），图像的每个通道的均值（mean）、方差（var）、缩放因子（gamma）和偏置（bias）
// 然后对图像进行批归一化处理，处理后的图像存储在新分配的内存中，并返回这个内存的指针。
float* MyBatchNorm(
    float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c) {
  // 分配输出内存空间
  float* out = (float*)malloc(h * w * c * sizeof(float));

  // 遍历每个通道
  for (auto c_ = 0; c_ < c; c_++) {
    // 获取当前通道的均值、方差、伽马值和偏置
    auto m = mean[c_];
    auto v = var[c_];
    auto gm = gamma[c_];
    auto bi = bias[c_];

    // 遍历高度和宽度
    for (auto hw = 0; hw < h * w; hw++) {
      // 获取当前像素点的值
      auto data = img[hw * c + c_];

      // 执行归一化和缩放操作
      auto data_ = (data - m) / sqrt(v + 1e-5);
      data_ = data_ * gm + bi;

      // 将处理后的值存储到输出数组
      out[hw * c + c_] = data_;
    }
  }

  // 释放输入数组和相关参数的内存
  free(img);
  // 返回处理后的数组
  return out;
}
