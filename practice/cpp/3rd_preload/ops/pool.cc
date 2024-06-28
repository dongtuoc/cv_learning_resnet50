#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 池化层在 preload 过程中，无需做任何事情，直接返回指针即可
float* MyMaxPoolPreLoad(float* img) { return img; }

float* MyAvgPoolPreLoad(float* img) { return img; }

// 最大池化操作
float* MyMaxPool(float* img) {
  // 设置池化操作的参数， resnet50 中的最大池化层只有一个，参数也是固定的，如下面两行所示。
  auto hi = 112, wi = 112, channel = 64;
  auto pad = 1, stride = 2, kernel = 3;
  // 计算输出的尺寸
  auto ho = (hi + 2 * pad - kernel) / stride + 1;
  auto wo = (wi + 2 * pad - kernel) / stride + 1;
  // 分配输出内存
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));

  // 遍历每个通道
  for (auto c_ = 0; c_ < channel; c_++) {
    // 遍历输出的高度
    for (auto ho_idx = 0; ho_idx < ho; ho_idx++) {
      int in_h_origin = ho_idx * stride - pad;
      // 遍历输出的宽度
      for (auto wo_idx = 0; wo_idx < wo; wo_idx++) {
        int in_w_origin = wo_idx * stride - pad;
        // 计算池化窗口的起始和结束位置
        auto filter_h_start = std::max(0, -in_h_origin);
        auto filter_w_start = std::max(0, -in_w_origin);
        auto filter_h_end = std::min(kernel, hi - in_h_origin);
        auto filter_w_end = std::min(kernel, wi - in_w_origin);
        float max_x = float(0);  // 初始化最大值
        // 在池化窗口内查找最大值，利用循环遍历的方式找到最大值
        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          auto hi_index = in_h_origin + kh_idx;
          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            auto wi_index = in_w_origin + kw_idx;
            auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
            max_x = std::max(in_data, max_x);  // 更新最大值
          }
        }
        // 存储池化结果
        out[ho_idx * wo * channel + wo_idx * channel + c_] = max_x;
      }
    }
  }
  free(img);
  return out;  // 返回池化后的输出
}

// 平均池化操作
float* MyAvgPool(float* img) {
  // 设置池化操作的参数，resnet50 中的平均池化层只有一个，参数也是固定的，如下面两行所示。
  auto hi = 7, wi = 7, channel = 2048;
  auto pad = 0, stride = 1, kernel = 7;
  // 计算输出的尺寸
  auto ho = (hi + 2 * pad - kernel) / stride + 1;
  auto wo = (wi + 2 * pad - kernel) / stride + 1;
  // 分配输出内存
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));

  // 遍历每个通道
  for (auto c_ = 0; c_ < channel; c_++) {
    // 遍历输出的高度
    for (auto ho_idx = 0; ho_idx < ho; ho_idx++) {
      int in_h_origin = ho_idx * stride - pad;
      // 遍历输出的宽度
      for (auto wo_idx = 0; wo_idx < wo; wo_idx++) {
        int in_w_origin = wo_idx * stride - pad;
        // 计算池化窗口的起始和结束位置
        auto filter_h_start = std::max(0, -in_h_origin);
        auto filter_w_start = std::max(0, -in_w_origin);
        auto filter_h_end = std::min(kernel, hi - in_h_origin);
        auto filter_w_end = std::min(kernel, wi - in_w_origin);
        float sum = float(0);  // 初始化求和
        // 在池化窗口内计算所有像素的累加值
        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          auto hi_index = in_h_origin + kh_idx;
          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            auto wi_index = in_w_origin + kw_idx;
            auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
            sum += in_data;  // 累加
          }
        }
        // 计算完累加值，因为最终要计算平均值，因此直接存储（累加值 / 窗口内像素数量 = 平均值）
        out[ho_idx * wo * channel + wo_idx * channel + c_] = sum / (kernel * kernel);
      }
    }
  }
  free(img);
  return out;  // 返回池化后的输出
}
