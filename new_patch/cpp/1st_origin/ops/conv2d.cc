#include "conv2d.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

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
#if DEBUG_SHOW
  printf("conv in: (%d, %d, %d)\n", hi, wi, ci);
#endif
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  float* out = (float*)malloc(ho * wo * co * sizeof(float));

  for (int co_idx = 0; co_idx < co; co_idx++) {
    for (int ho_idx = 0; ho_idx < ho; ho_idx++) {
      const int in_h_origin = ho_idx * stride - pad;
      for (int wo_idx = 0; wo_idx < wo; wo_idx++) {
        const int in_w_origin = wo_idx * stride - pad;
        const int filter_h_start = std::max(0, -in_h_origin);
        const int filter_w_start = std::max(0, -in_w_origin);
        const int filter_h_end = std::min(kernel, hi - in_h_origin);
        const int filter_w_end = std::min(kernel, wi - in_w_origin);
        float acc = 0;
        for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {
          const int hi_index = in_h_origin + kh_idx;
          for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {
            const int wi_index = in_w_origin + kw_idx;
            for (int ci_ = 0; ci_ < ci; ci_++) {
              auto in_data = img[hi_index * wi * ci + wi_index * ci + ci_];
              auto weight_data =
                  weight[co_idx * kernel * kernel * ci + kh_idx * kernel * ci + kw_idx * ci + ci_];
              acc += in_data * weight_data;
            }
          }
        }
        out[ho_idx * wo * co + wo_idx * co + co_idx] = acc;
      }
    }
  }

  if (is_free_img) {
    //  free(img);
  }
  free(weight);
#if DEBUG_SHOW
  printf("conv out: (%d, %d, %d)\n", ho, wo, co);
#endif
  return out;
}
