#include "fc.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* my_fc(float* img, float* weight, float* bias) {
#if DEBUG_SHOW
  printf("fc in: (1000, 2048)\n");
  printf("fc out: (1000)\n");
#endif
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
  free(weight);
  free(bias);
  return out;
}
