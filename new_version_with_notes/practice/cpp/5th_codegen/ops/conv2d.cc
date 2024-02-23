#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "common.h"

void MyConv2dPreLoad(void* img_in,
                     void* img_out,
                     float* weight,
                     int hi,
                     int wi,
                     int& ho,
                     int& wo,
                     int ci,
                     int co,
                     int kernel,
                     int stride,
                     int pad) {
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;
  return;
}

int conv_idx = 0;
void MyConv2d(void* img_in,
              void* img_out,
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
              bool First) {
  ho = (hi + 2 * pad - kernel) / stride + 1;
  wo = (wi + 2 * pad - kernel) / stride + 1;

  std::ostringstream imp_os;
  std::ostringstream dec_os;
  dec_os << "void MyConv2d_" << conv_idx << "(void* img_in, void* img_out, float* weight);\n";
  imp_os << "#include <algorithm>\n";
  imp_os << "#include \"func.h\"\n";
  imp_os << "void MyConv2d_" << conv_idx << "(void* img_in, void* img_out, float* weight) {\n";
  imp_os << "  float* img = (float*)img_in;\n";
  imp_os << "  float* out = (float*)img_out;\n";

  imp_os << "  for (int ho_idx = 0; ho_idx < " << ho << "; ho_idx++) {\n";
  imp_os << "    const int in_h_origin = ho_idx * " << stride << " - " << pad << ";\n";
  imp_os << "    for (int wo_idx = 0; wo_idx < " << wo << "; wo_idx++) {\n";
  imp_os << "      const int in_w_origin = wo_idx * " << stride << " - " << pad << ";\n";
  imp_os << "      const int filter_h_start = std::max(0, -in_h_origin);\n";
  imp_os << "      const int filter_w_start = std::max(0, -in_w_origin);\n";
  imp_os << "      const int filter_h_end = std::min(" << kernel << ", " << hi
         << " - in_h_origin);\n";
  imp_os << "      const int filter_w_end = std::min(" << kernel << ", " << wi
         << " - in_w_origin);\n";
  imp_os << "      for (int co_idx = 0; co_idx < " << co << "; co_idx++) {\n";
  imp_os << "        register float acc = 0;\n";
  if (conv_idx == 0) {
    imp_os << "        for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {\n";
    imp_os << "          const int hi_index = in_h_origin + kh_idx;\n";
    imp_os << "          for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {\n";
    imp_os << "            const int wi_index = in_w_origin + kw_idx;\n";
    imp_os << "            for (int ci_ = 0; ci_ < 3; ci_++) {\n";
    imp_os << "              auto in_data = img[hi_index * 224 * 3 + wi_index * 3 + ci_];\n";
    imp_os << "              auto weight_data = weight[co_idx * 49 * 3 + kh_idx * 7 * 3 + kw_idx * "
              "3 + ci_];\n";
    imp_os << "              acc += in_data * weight_data;\n";
    imp_os << "            }\n";
    imp_os << "          }\n";
    imp_os << "        }\n";
  } else {
    imp_os << "        // use avx2 vec inst to optimize Mul-add operation\n";
    imp_os << "        const int vec_size = 8;\n";
    imp_os << "        for (int kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {\n";
    imp_os << "          const register int hi_index = in_h_origin + kh_idx;\n";
    imp_os << "          for (int kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {\n";
    imp_os << "            const register int wi_index = in_w_origin + kw_idx;\n";
    imp_os << "            // Load input and weight data into vectors\n";
    imp_os << "            __m256 in_vec, weight_vec;\n";
    imp_os << "            for (int ci_ = 0; ci_ < " << ci << "; ci_ += vec_size) {\n";
    imp_os << "              in_vec = _mm256_loadu_ps(&img[hi_index * " << wi * ci
           << " + wi_index * " << ci << " + ci_]);\n";
    imp_os << "              weight_vec = _mm256_loadu_ps(&weight[co_idx * " << kernel * kernel * ci
           << " +\n";
    imp_os << "                                                   kh_idx * " << kernel * ci
           << " + kw_idx * " << ci << " + ci_]);\n";
    imp_os << "              in_vec = _mm256_mul_ps(in_vec, weight_vec);\n";
    imp_os << "              // Add the elements of the accumulator vector and store the result\n";
    imp_os << "              acc += avx2_sum(in_vec);\n";
    imp_os << "            }\n";
    imp_os << "          }\n";
    imp_os << "        }\n";
  }
  imp_os << "        out[ho_idx * " << wo * co << " + wo_idx * " << co << " + co_idx] = acc;\n";
  imp_os << "      }\n";
  imp_os << "    }\n";
  imp_os << "  }\n";
  imp_os << "}\n";
  const std::string fi = "codegen/conv_" + std::to_string(conv_idx) + ".cc";
  const std::string fd = "codegen/conv_" + std::to_string(conv_idx) + ".h";
  write(fi, imp_os);
  write(fd, dec_os);
  return;
}
