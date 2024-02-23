#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

inline float avx2_sum(__m256 in_vec) {
  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));
  in_vec = _mm256_hadd_ps(in_vec, in_vec);
  float sum0 = _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));
  return sum0;
}

void MyFCPreLoad(void* img_in, void* img_out, float* weight, float* bias) { return; }

void MyFC(void* img_in, void* img_out, float* weight, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
  for (int i = 0; i < 1000; i++) {
    float sum_x = float(0);
    const int vec_size = 8;
    __m256 l_vec, weight_vec;
    for (int j = 0; j < 2048; j += vec_size) {
      l_vec = _mm256_loadu_ps(&img[j]);
      weight_vec = _mm256_loadu_ps(&weight[i * 2048 + j]);
      l_vec = _mm256_mul_ps(l_vec, weight_vec);
      // Add the elements of the accumulator vector and store the result
      sum_x += avx2_sum(l_vec);
    }
    out[i] = sum_x + bias[i];
  }
  return;
}
