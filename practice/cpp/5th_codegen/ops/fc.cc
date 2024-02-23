#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "common.h"

inline float avx2_sum(__m256 in_vec) {
  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));
  in_vec = _mm256_hadd_ps(in_vec, in_vec);
  float sum0 = _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));
  return sum0;
}

void MyFCPreLoad(void* img_in, void* img_out, float* weight, float* bias) { return; }

void MyFC(void* img_in, void* img_out, float* weight, float* bias) {
#if CODE_GEN
  std::ostringstream hfc_os;
  std::ostringstream fc_os;
  hfc_os << "void MyFC(void* img_in, void* img_out, float* weight, float* "
            "bias);\n";
  fc_os << "#include \"func.h\"\n";
  fc_os << "void MyFC(void* img_in, void* img_out, float* weight, float* "
           "bias) {\n";
  fc_os << "  float* img = (float*)img_in;\n";
  fc_os << "  float* out = (float*)img_out;\n";
  fc_os << "  for (int outer = 0; outer < 1000; ++outer) {\n";
  fc_os << "    float sum_x = float(0);\n";
  fc_os << "    const int vec_size = 8;\n";
  fc_os << "    __m256 l_vec, weight_vec;\n";
  fc_os << "    for (int inner = 0; inner < 2048; inner += vec_size) {\n";
  fc_os << "      l_vec = _mm256_loadu_ps(&img[inner]);\n";
  fc_os << "      weight_vec = _mm256_loadu_ps(&weight[outer * 2048 + inner]);\n";
  fc_os << "      l_vec = _mm256_mul_ps(l_vec, weight_vec);\n";
  fc_os << "      sum_x += avx2_sum(l_vec);\n";
  fc_os << "    }\n";
  fc_os << "    out[outer] = sum_x + bias[outer];\n";
  fc_os << "  }\n";
  fc_os << "}\n";
  write("codegen/fc.cc", fc_os);
  write("codegen/fc.h", hfc_os);
  return;
#endif

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
