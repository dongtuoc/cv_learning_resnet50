#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "common.h"

void MyBatchNormPreLoad(void* img_in,
                        void* img_out,
                        float* mean,
                        float* var,
                        float* gamma,
                        float* bias,
                        int h,
                        int w,
                        int c) {
  return;
}

int bn_cnt = 0;
void MyBatchNorm(void* img_in,
                 void* img_out,
                 float* mean,
                 float* var,
                 float* gamma,
                 float* bias,
                 int h,
                 int w,
                 int c) {
  std::ostringstream hbn_os;
  std::ostringstream bn_os;
  hbn_os << "void MyBatchNorm_" << bn_cnt << "(\n";
  hbn_os
      << "    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias);\n";
  if (bn_cnt == 0) {
    bn_os << "#include <cmath>\n";
  }
  bn_os << "void MyBatchNorm_" << bn_cnt << "(\n";
  bn_os
      << "    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {\n";
  bn_os << "  float* img = (float*)img_in;\n";
  bn_os << "  float* out = (float*)img_out;\n";
  bn_os << "  for (auto c_ = 0; c_ < " << c << "; c_++) {\n";
  bn_os << "    auto m = mean[c_];\n";
  bn_os << "    auto v = var[c_];\n";
  bn_os << "    auto gm = gamma[c_];\n";
  bn_os << "    auto bi = bias[c_];\n";
  bn_os << "    float div_res = sqrt(v + 1e-5);\n";
  bn_os << "    for (auto hw = 0; hw < " << h * w << "; hw++) {\n";
  bn_os << "      auto data = img[hw * " << c << " + c_];\n";
  bn_os << "      auto data_ = (data - m) / div_res;\n";
  bn_os << "      data_ = data_ * gm + bi;\n";
  bn_os << "      out[hw * " << c << " + c_] = data_;\n";
  bn_os << "    }\n";
  bn_os << "  }\n";
  bn_os << "}\n";

  write("codegen/bn.cc", bn_os);
  write("codegen/bn.h", hbn_os);
  return;
}
