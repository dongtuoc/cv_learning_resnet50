#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 定义预加载版本的批量归一化函数，此版本不执行任何操作
void MyBatchNormPreLoad(void* img_in,
                        void* img_out,
                        float* mean,
                        float* var,
                        float* gamma,
                        float* bias,
                        int h,
                        int w,
                        int c) {
  return;
}

// 批量归一化函数，执行实际的归一化操作
void MyBatchNorm(void* img_in,
                 void* img_out,
                 float* mean,
                 float* var,
                 float* gamma,
                 float* bias,
                 int h,
                 int w,
                 int c) {
  // 类型转换输入和输出为 float 指针
  float* img = (float*)img_in;
  float* out = (float*)img_out;

  // 遍历每个通道
  for (auto c_ = 0; c_ < c; c_++) {
    // 获取当前通道的均值、方差、比例因子和偏置
    auto m = mean[c_];
    auto v = var[c_];
    auto gm = gamma[c_];
    auto bi = bias[c_];

    // 遍历每个像素
    for (auto hw = 0; hw < h * w; hw++) {
      // 读取当前像素的值
      auto data = img[hw * c + c_];
      // 执行归一化操作：(data - mean) / sqrt(variance + epsilon)
      auto data_ = (data - m) / sqrt(v + 1e-5);
      // 应用比例因子和偏置
      data_ = data_ * gm + bi;
      // 将归一化结果写入输出数组
      out[hw * c + c_] = data_;
    }
  }
}