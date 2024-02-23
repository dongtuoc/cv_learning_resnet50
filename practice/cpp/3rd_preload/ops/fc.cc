#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* MyFCPreLoad(float* img, float* weight, float* bias) { return img; }

float* MyFC(float* img, float* weight, float* bias) {
  float* out = (float*)malloc(1000 * sizeof(float));
  for (int i = 0; i < 1000; i++) {
    float sum_x = float(0);
    for (int j = 0; j < 2048; j++) {
      auto l = img[j];
      auto r = weight[i * 2048 + j];
      sum_x += l * r;
    }
    out[i] = sum_x + bias[i];
  }
  free(img);
  return out;
}
