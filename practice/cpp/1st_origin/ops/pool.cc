#include "pool.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* my_max_pool(float* img) {
  auto hi = 112;
  auto wi = 112;
  auto channel = 64;
  auto pad = 1;
  auto stride = 2;
  auto kernel = 3;
  auto ho = (hi + 2 * pad - kernel) / stride + 1;
  auto wo = (wi + 2 * pad - kernel) / stride + 1;
#if DEBUG_SHOW
  printf("maxpool in: (%d, %d, %d)\n", hi, wi, channel);
#endif
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));

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
  free(img);
#if DEBUG_SHOW
  printf("maxpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
  return out;
}

float* my_avg_pool(float* img) {
  auto hi = 7;
  auto wi = 7;
  auto channel = 2048;
  auto pad = 0;
  auto stride = 1;
  auto kernel = 7;
  auto ho = (hi + 2 * pad - kernel) / stride + 1;
  auto wo = (wi + 2 * pad - kernel) / stride + 1;
  float* out = (float*)malloc(ho * wo * channel * sizeof(float));
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
        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          auto hi_index = in_h_origin + kh_idx;
          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            auto wi_index = in_w_origin + kw_idx;
            auto in_data = img[hi_index * wi * channel + wi_index * channel + c_];
            sum += in_data;
          }
        }
        out[ho_idx * wo * channel + wo_idx * channel + c_] = sum / (kernel * kernel);
      }
    }
  }
  free(img);
#if DEBUG_SHOW
  printf("avgpool out: (%d, %d, %d)\n", ho, wo, channel);
#endif
  return out;
}
