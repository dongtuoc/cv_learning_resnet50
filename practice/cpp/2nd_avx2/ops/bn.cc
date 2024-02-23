#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* my_bn(float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c) {
  float* out = (float*)malloc(h * w * c * sizeof(float));
  for (auto c_ = 0; c_ < c; c_++) {
    auto m = mean[c_];
    auto v = var[c_];
    auto gm = gamma[c_];
    auto bi = bias[c_];
    for (auto hw = 0; hw < h * w; hw++) {
      auto data = img[hw * c + c_];
      auto data_ = (data - m) / sqrt(v + 1e-5);
      data_ = data_ * gm + bi;
      out[hw * c + c_] = data_;
    }
  }
  free(img);
  free(mean);
  free(var);
  free(gamma);
  free(bias);
  return out;
}
