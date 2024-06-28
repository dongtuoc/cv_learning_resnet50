#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "common.h"

void MyFCPreLoad(void* img_in, void* img_out, float* weight, float* bias) { return; }

void MyFC(void* img_in, void* img_out, float* weight, float* bias) {
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
}
