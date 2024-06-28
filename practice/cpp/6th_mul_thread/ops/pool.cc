#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "common.h"

void MyMaxPoolPreLoad(void* img_in, void* img_out) { return; }

void MyMaxPool(void* img_in, void* img_out) {
  std::ostringstream hmaxpool_os;
  std::ostringstream maxpool_os;
  hmaxpool_os << "void MyMaxPool(void* img_in, void* img_out);\n";
  maxpool_os << "#include <algorithm>\n";
  maxpool_os << "#include \"func.h\"\n";
  maxpool_os << "void MyMaxPool(void* img_in, void* img_out) {\n";
  maxpool_os << "  float* img = (float*)img_in;\n";
  maxpool_os << "  float* out = (float*)img_out;\n";
  maxpool_os << "  for (auto c_ = 0; c_ < 64; c_++) {\n";
  maxpool_os << "    for (auto ho_idx = 0; ho_idx < 56; ho_idx++) {\n";
  maxpool_os << "      int in_h_origin = ho_idx * 2 - 1;\n";
  maxpool_os << "      for (auto wo_idx = 0; wo_idx < 56; wo_idx++) {\n";
  maxpool_os << "        int in_w_origin = wo_idx * 2 - 1;\n";
  maxpool_os << "        auto filter_h_start = std::max(0, -in_h_origin);\n";
  maxpool_os << "        auto filter_w_start = std::max(0, -in_w_origin);\n";
  maxpool_os << "        auto filter_h_end = std::min(3, 112 - in_h_origin);\n";
  maxpool_os << "        auto filter_w_end = std::min(3, 112 - in_w_origin);\n";
  maxpool_os << "        float max_x = float(0);\n";
  maxpool_os << "        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {\n";
  maxpool_os << "          auto hi_index = in_h_origin + kh_idx;\n";
  maxpool_os << "          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {\n";
  maxpool_os << "            auto wi_index = in_w_origin + kw_idx;\n";
  maxpool_os << "            auto in_data = img[hi_index * 112 * 64 + wi_index * 64 + c_];\n";
  maxpool_os << "            max_x = std::max(in_data, max_x);\n";
  maxpool_os << "          }\n";
  maxpool_os << "        }\n";
  maxpool_os << "        out[ho_idx * 56 * 64 + wo_idx * 64 + c_] = max_x;\n";
  maxpool_os << "      }\n";
  maxpool_os << "    }\n";
  maxpool_os << "  }\n";
  maxpool_os << "}\n";

  write("codegen/maxpool.h", hmaxpool_os);
  write("codegen/maxpool.cc", maxpool_os);
  return;
}

void MyAvgPoolPreLoad(void* img_in, void* img_out) { return; }

void MyAvgPool(void* img_in, void* img_out) {
  std::ostringstream havgpool_os;
  std::ostringstream avgpool_os;
  havgpool_os << "void MyAvgPool(void* img_in, void* img_out);\n";
  avgpool_os << "#include <algorithm>\n";
  avgpool_os << "#include \"func.h\"\n";
  avgpool_os << "void MyAvgPool(void* img_in, void* img_out) {\n";
  avgpool_os << "  float* img = (float*)img_in;\n";
  avgpool_os << "  float* out = (float*)img_out;\n";
  avgpool_os << "  for (auto c_ = 0; c_ < 2048; c_++) {\n";
  avgpool_os << "    for (auto ho_idx = 0; ho_idx < 1; ho_idx++) {\n";
  avgpool_os << "      int in_h_origin = ho_idx * 1 - 0;\n";
  avgpool_os << "      for (auto wo_idx = 0; wo_idx < 1; wo_idx++) {\n";
  avgpool_os << "        int in_w_origin = wo_idx * 1 - 0;\n";
  avgpool_os << "        auto filter_h_start = std::max(0, -in_h_origin);\n";
  avgpool_os << "        auto filter_w_start = std::max(0, -in_w_origin);\n";
  avgpool_os << "        auto filter_h_end = std::min(7, 7 - in_h_origin);\n";
  avgpool_os << "        auto filter_w_end = std::min(7, 7 - in_w_origin);\n";
  avgpool_os << "        float sum = float(0);\n";
  avgpool_os << "        int k_size = (filter_h_end - filter_h_start) * (filter_w_end - "
                "filter_w_start);\n";
  avgpool_os << "        for (auto kh_idx = filter_h_start; kh_idx < filter_h_end; kh_idx++) {\n";
  avgpool_os << "          auto hi_index = in_h_origin + kh_idx;\n";
  avgpool_os << "          for (auto kw_idx = filter_w_start; kw_idx < filter_w_end; kw_idx++) {\n";
  avgpool_os << "            auto wi_index = in_w_origin + kw_idx;\n";
  avgpool_os << "            auto in_data = img[hi_index * 7 * 2048 + wi_index * 2048 + c_];\n";
  avgpool_os << "            sum += in_data;\n";
  avgpool_os << "          }\n";
  avgpool_os << "        }\n";
  avgpool_os << "        out[ho_idx * 1 * 2048 + wo_idx * 2048 + c_] = sum / k_size;\n";
  avgpool_os << "      }\n";
  avgpool_os << "    }\n";
  avgpool_os << "  }\n";
  avgpool_os << "}\n";

  write("codegen/avgpool.h", havgpool_os);
  write("codegen/avgpool.cc", avgpool_os);
  return;
}
