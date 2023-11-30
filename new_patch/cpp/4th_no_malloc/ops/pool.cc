#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

void MyMaxPoolPreLoad(void* img_in, void* img_out) { return; }

void MyMaxPool(void* img_in, void* img_out) {
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
}

void MyAvgPoolPreLoad(void* img_in, void* img_out) { return; }

void MyAvgPool(void* img_in, void* img_out) {
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
}
