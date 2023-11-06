#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
inline float avx2_sum(__m256 in_vec) {
  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));
  in_vec = _mm256_hadd_ps(in_vec, in_vec);
  return _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));
}
static void MyConv2dInfer_0(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_0(void* img_in, void* img_out) {
  MyConv2dInfer_0(img_in, img_out, (float*)0x55b0689c8e30);
}
static void MyBatchNormInfer_0(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_0(void* in_data, void* out_data) {
  MyBatchNormInfer_0(in_data, out_data, (float*)0x55b0689d2370, (float*)0x55b0689d2480,
                     (float*)0x55b068989c30, (float*)0x55b0689d2260);
}
static void InferLayerRelu_0(void* img_in) {
  const auto& len = 802816;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyMaxPoolInfer(void* img_in, void* img_out) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerMaxPool(void* in_data, void* out_data) { MyMaxPoolInfer(in_data, out_data); }
static void MyConv2dInfer_1(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_1(void* img_in, void* img_out) {
  MyConv2dInfer_1(img_in, img_out, (float*)0x55b0689d26b0);
}
static void MyBatchNormInfer_1(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_1(void* in_data, void* out_data) {
  MyBatchNormInfer_1(in_data, out_data, (float*)0x55b0689d6a60, (float*)0x55b0689d6b70,
                     (float*)0x55b0689d6840, (float*)0x55b0689d6950);
}
static void InferLayerRelu_1(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_2(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_2(void* img_in, void* img_out) {
  MyConv2dInfer_2(img_in, img_out, (float*)0x7f1acd3fa010);
}
static void MyBatchNormInfer_2(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_2(void* in_data, void* out_data) {
  MyBatchNormInfer_2(in_data, out_data, (float*)0x55b0689d6ec0, (float*)0x55b0689d6fd0,
                     (float*)0x55b0689d6ca0, (float*)0x55b0689d6db0);
}
static void InferLayerRelu_2(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_3(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_3(void* img_in, void* img_out) {
  MyConv2dInfer_3(img_in, img_out, (float*)0x55b0689d7100);
}
static void MyBatchNormInfer_3(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_3(void* in_data, void* out_data) {
  MyBatchNormInfer_3(in_data, out_data, (float*)0x55b0689e7520, (float*)0x55b0689e7930,
                     (float*)0x55b0684dc800, (float*)0x55b0689e7110);
}
static void MyConv2dInfer_4(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_4(void* img_in, void* img_out) {
  MyConv2dInfer_4(img_in, img_out, (float*)0x55b0689e7e00);
}
static void MyBatchNormInfer_4(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_4(void* in_data, void* out_data) {
  MyBatchNormInfer_4(in_data, out_data, (float*)0x55b0689f8910, (float*)0x55b0689f8d20,
                     (float*)0x55b0689f80f0, (float*)0x55b0689f8500);
}
static void InferLayerAdd_0(float* l, float* r, float* out) {
  const auto& len = 802816;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_3(void* img_in) {
  const auto& len = 802816;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_0(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_1(in_data, out_data);
  InferLayerBatchNorm_1(out_data, temp_data);
  InferLayerRelu_1(temp_data);
  InferLayerConv2d_2(temp_data, out_data);
  InferLayerBatchNorm_2(out_data, temp_data);
  InferLayerRelu_2(temp_data);
  InferLayerConv2d_3(temp_data, out_data);
  InferLayerBatchNorm_3(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerConv2d_4(in_data, out_data);
  InferLayerBatchNorm_4(out_data, in_data);
  auto short_cut_out = in_data;
  InferLayerAdd_0((float*)bn_out, (float*)short_cut_out, (float*)out_data);
  InferLayerRelu_3(temp_data);
}
static void MyConv2dInfer_5(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_5(void* img_in, void* img_out) {
  MyConv2dInfer_5(img_in, img_out, (float*)0x55b0689f9130);
}
static void MyBatchNormInfer_5(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_5(void* in_data, void* out_data) {
  MyBatchNormInfer_5(in_data, out_data, (float*)0x55b068a09360, (float*)0x55b068a09470,
                     (float*)0x55b068a09140, (float*)0x55b068a09250);
}
static void InferLayerRelu_4(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_6(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_6(void* img_in, void* img_out) {
  MyConv2dInfer_6(img_in, img_out, (float*)0x7f1acd3d5010);
}
static void MyBatchNormInfer_6(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_6(void* in_data, void* out_data) {
  MyBatchNormInfer_6(in_data, out_data, (float*)0x55b068a097c0, (float*)0x55b068a098d0,
                     (float*)0x55b068a095a0, (float*)0x55b068a096b0);
}
static void InferLayerRelu_5(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_7(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_7(void* img_in, void* img_out) {
  MyConv2dInfer_7(img_in, img_out, (float*)0x55b068a09a00);
}
static void MyBatchNormInfer_7(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_7(void* in_data, void* out_data) {
  MyBatchNormInfer_7(in_data, out_data, (float*)0x55b068a1a230, (float*)0x55b068a1a640,
                     (float*)0x55b068a19a10, (float*)0x55b068a19e20);
}
static void InferLayerAdd_1(float* l, float* r, float* out) {
  const auto& len = 802816;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_6(void* img_in) {
  const auto& len = 802816;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_1(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_5(in_data, out_data);
  InferLayerBatchNorm_5(out_data, temp_data);
  InferLayerRelu_4(temp_data);
  InferLayerConv2d_6(temp_data, out_data);
  InferLayerBatchNorm_6(out_data, temp_data);
  InferLayerRelu_5(temp_data);
  InferLayerConv2d_7(temp_data, out_data);
  InferLayerBatchNorm_7(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_1((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_6(out_data);
}
static void MyConv2dInfer_8(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_8(void* img_in, void* img_out) {
  MyConv2dInfer_8(img_in, img_out, (float*)0x55b068a1aa70);
}
static void MyBatchNormInfer_8(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_8(void* in_data, void* out_data) {
  MyBatchNormInfer_8(in_data, out_data, (float*)0x55b068a2aca0, (float*)0x55b068a2adb0,
                     (float*)0x55b068a2aa80, (float*)0x55b068a2ab90);
}
static void InferLayerRelu_7(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_9(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_9(void* img_in, void* img_out) {
  MyConv2dInfer_9(img_in, img_out, (float*)0x7f1acd3b0010);
}
static void MyBatchNormInfer_9(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_9(void* in_data, void* out_data) {
  MyBatchNormInfer_9(in_data, out_data, (float*)0x55b068a2b100, (float*)0x55b068a2b210,
                     (float*)0x55b068a2aee0, (float*)0x55b068a2aff0);
}
static void InferLayerRelu_8(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_10(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_10(void* img_in, void* img_out) {
  MyConv2dInfer_10(img_in, img_out, (float*)0x55b068a2b340);
}
static void MyBatchNormInfer_10(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_10(void* in_data, void* out_data) {
  MyBatchNormInfer_10(in_data, out_data, (float*)0x55b068a3bb70, (float*)0x55b068a3bf80,
                      (float*)0x55b068a3b350, (float*)0x55b068a3b760);
}
static void InferLayerAdd_2(float* l, float* r, float* out) {
  const auto& len = 802816;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_9(void* img_in) {
  const auto& len = 802816;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_2(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_8(in_data, out_data);
  InferLayerBatchNorm_8(out_data, temp_data);
  InferLayerRelu_7(temp_data);
  InferLayerConv2d_9(temp_data, out_data);
  InferLayerBatchNorm_9(out_data, temp_data);
  InferLayerRelu_8(temp_data);
  InferLayerConv2d_10(temp_data, out_data);
  InferLayerBatchNorm_10(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_2((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_9(out_data);
}
static void MyConv2dInfer_11(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_11(void* img_in, void* img_out) {
  MyConv2dInfer_11(img_in, img_out, (float*)0x7f1acd38f010);
}
static void MyBatchNormInfer_11(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_11(void* in_data, void* out_data) {
  MyBatchNormInfer_11(in_data, out_data, (float*)0x55b068a3c7d0, (float*)0x55b068a3c9e0,
                      (float*)0x55b068a3c3b0, (float*)0x55b068a3c5c0);
}
static void InferLayerRelu_10(void* img_in) {
  const auto& len = 401408;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_12(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_12(void* img_in, void* img_out) {
  MyConv2dInfer_12(img_in, img_out, (float*)0x7f1acd2fe010);
}
static void MyBatchNormInfer_12(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_12(void* in_data, void* out_data) {
  MyBatchNormInfer_12(in_data, out_data, (float*)0x55b068a3d030, (float*)0x55b068a3d240,
                      (float*)0x55b068a3cc10, (float*)0x55b068a3ce20);
}
static void InferLayerRelu_11(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_13(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_13(void* img_in, void* img_out) {
  MyConv2dInfer_13(img_in, img_out, (float*)0x7f1acd2bd010);
}
static void MyBatchNormInfer_13(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_13(void* in_data, void* out_data) {
  MyBatchNormInfer_13(in_data, out_data, (float*)0x55b068a3e490, (float*)0x55b068a3eca0,
                      (float*)0x55b068a3d470, (float*)0x55b068a3dc80);
}
static void MyConv2dInfer_14(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_14(void* img_in, void* img_out) {
  MyConv2dInfer_14(img_in, img_out, (float*)0x7f1acd23c010);
}
static void MyBatchNormInfer_14(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_14(void* in_data, void* out_data) {
  MyBatchNormInfer_14(in_data, out_data, (float*)0x55b068a40510, (float*)0x55b068a40d20,
                      (float*)0x55b068a3f4f0, (float*)0x55b068a3fd00);
}
static void InferLayerAdd_3(float* l, float* r, float* out) {
  const auto& len = 401408;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_12(void* img_in) {
  const auto& len = 401408;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_3(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_11(in_data, out_data);
  InferLayerBatchNorm_11(out_data, temp_data);
  InferLayerRelu_10(temp_data);
  InferLayerConv2d_12(temp_data, out_data);
  InferLayerBatchNorm_12(out_data, temp_data);
  InferLayerRelu_11(temp_data);
  InferLayerConv2d_13(temp_data, out_data);
  InferLayerBatchNorm_13(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerConv2d_14(in_data, out_data);
  InferLayerBatchNorm_14(out_data, in_data);
  auto short_cut_out = in_data;
  InferLayerAdd_3((float*)bn_out, (float*)short_cut_out, (float*)out_data);
  InferLayerRelu_12(temp_data);
}
static void MyConv2dInfer_15(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_15(void* img_in, void* img_out) {
  MyConv2dInfer_15(img_in, img_out, (float*)0x7f1acd1fb010);
}
static void MyBatchNormInfer_15(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_15(void* in_data, void* out_data) {
  MyBatchNormInfer_15(in_data, out_data, (float*)0x55b068a41950, (float*)0x55b068a41b60,
                      (float*)0x55b068a41530, (float*)0x55b068a41740);
}
static void InferLayerRelu_13(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_16(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_16(void* img_in, void* img_out) {
  MyConv2dInfer_16(img_in, img_out, (float*)0x7f1acd16a010);
}
static void MyBatchNormInfer_16(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_16(void* in_data, void* out_data) {
  MyBatchNormInfer_16(in_data, out_data, (float*)0x55b068a421b0, (float*)0x55b068a423c0,
                      (float*)0x55b068a41d90, (float*)0x55b068a41fa0);
}
static void InferLayerRelu_14(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_17(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_17(void* img_in, void* img_out) {
  MyConv2dInfer_17(img_in, img_out, (float*)0x7f1acd129010);
}
static void MyBatchNormInfer_17(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_17(void* in_data, void* out_data) {
  MyBatchNormInfer_17(in_data, out_data, (float*)0x55b068a43610, (float*)0x55b068a43e20,
                      (float*)0x55b068a425f0, (float*)0x55b068a42e00);
}
static void InferLayerAdd_4(float* l, float* r, float* out) {
  const auto& len = 401408;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_15(void* img_in) {
  const auto& len = 401408;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_4(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_15(in_data, out_data);
  InferLayerBatchNorm_15(out_data, temp_data);
  InferLayerRelu_13(temp_data);
  InferLayerConv2d_16(temp_data, out_data);
  InferLayerBatchNorm_16(out_data, temp_data);
  InferLayerRelu_14(temp_data);
  InferLayerConv2d_17(temp_data, out_data);
  InferLayerBatchNorm_17(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_4((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_15(out_data);
}
static void MyConv2dInfer_18(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_18(void* img_in, void* img_out) {
  MyConv2dInfer_18(img_in, img_out, (float*)0x7f1acd0e8010);
}
static void MyBatchNormInfer_18(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_18(void* in_data, void* out_data) {
  MyBatchNormInfer_18(in_data, out_data, (float*)0x55b068a44a70, (float*)0x55b068a44c80,
                      (float*)0x55b068a44650, (float*)0x55b068a44860);
}
static void InferLayerRelu_16(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_19(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_19(void* img_in, void* img_out) {
  MyConv2dInfer_19(img_in, img_out, (float*)0x7f1acd057010);
}
static void MyBatchNormInfer_19(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_19(void* in_data, void* out_data) {
  MyBatchNormInfer_19(in_data, out_data, (float*)0x55b068a452d0, (float*)0x55b068a454e0,
                      (float*)0x55b068a44eb0, (float*)0x55b068a450c0);
}
static void InferLayerRelu_17(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_20(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_20(void* img_in, void* img_out) {
  MyConv2dInfer_20(img_in, img_out, (float*)0x7f1acd016010);
}
static void MyBatchNormInfer_20(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_20(void* in_data, void* out_data) {
  MyBatchNormInfer_20(in_data, out_data, (float*)0x55b068a46730, (float*)0x55b068a46f40,
                      (float*)0x55b068a45710, (float*)0x55b068a45f20);
}
static void InferLayerAdd_5(float* l, float* r, float* out) {
  const auto& len = 401408;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_18(void* img_in) {
  const auto& len = 401408;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_5(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_18(in_data, out_data);
  InferLayerBatchNorm_18(out_data, temp_data);
  InferLayerRelu_16(temp_data);
  InferLayerConv2d_19(temp_data, out_data);
  InferLayerBatchNorm_19(out_data, temp_data);
  InferLayerRelu_17(temp_data);
  InferLayerConv2d_20(temp_data, out_data);
  InferLayerBatchNorm_20(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_5((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_18(out_data);
}
static void MyConv2dInfer_21(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_21(void* img_in, void* img_out) {
  MyConv2dInfer_21(img_in, img_out, (float*)0x7f1accfd5010);
}
static void MyBatchNormInfer_21(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_21(void* in_data, void* out_data) {
  MyBatchNormInfer_21(in_data, out_data, (float*)0x55b068a47b90, (float*)0x55b068a47da0,
                      (float*)0x55b068a47770, (float*)0x55b068a47980);
}
static void InferLayerRelu_19(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_22(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_22(void* img_in, void* img_out) {
  MyConv2dInfer_22(img_in, img_out, (float*)0x7f1accf44010);
}
static void MyBatchNormInfer_22(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_22(void* in_data, void* out_data) {
  MyBatchNormInfer_22(in_data, out_data, (float*)0x55b068a483f0, (float*)0x55b068a48600,
                      (float*)0x55b068a47fd0, (float*)0x55b068a481e0);
}
static void InferLayerRelu_20(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_23(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_23(void* img_in, void* img_out) {
  MyConv2dInfer_23(img_in, img_out, (float*)0x7f1accf03010);
}
static void MyBatchNormInfer_23(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_23(void* in_data, void* out_data) {
  MyBatchNormInfer_23(in_data, out_data, (float*)0x55b068a49850, (float*)0x55b068a4a060,
                      (float*)0x55b068a48830, (float*)0x55b068a49040);
}
static void InferLayerAdd_6(float* l, float* r, float* out) {
  const auto& len = 401408;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_21(void* img_in) {
  const auto& len = 401408;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_6(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_21(in_data, out_data);
  InferLayerBatchNorm_21(out_data, temp_data);
  InferLayerRelu_19(temp_data);
  InferLayerConv2d_22(temp_data, out_data);
  InferLayerBatchNorm_22(out_data, temp_data);
  InferLayerRelu_20(temp_data);
  InferLayerConv2d_23(temp_data, out_data);
  InferLayerBatchNorm_23(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_6((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_21(out_data);
}
static void MyConv2dInfer_24(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_24(void* img_in, void* img_out) {
  MyConv2dInfer_24(img_in, img_out, (float*)0x7f1acce82010);
}
static void MyBatchNormInfer_24(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_24(void* in_data, void* out_data) {
  MyBatchNormInfer_24(in_data, out_data, (float*)0x55b068a4b0b0, (float*)0x55b068a4b4c0,
                      (float*)0x55b068a4a890, (float*)0x55b068a4aca0);
}
static void InferLayerRelu_22(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_25(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_25(void* img_in, void* img_out) {
  MyConv2dInfer_25(img_in, img_out, (float*)0x7f1accc41010);
}
static void MyBatchNormInfer_25(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_25(void* in_data, void* out_data) {
  MyBatchNormInfer_25(in_data, out_data, (float*)0x55b068a4c110, (float*)0x55b068a4c520,
                      (float*)0x55b068a4b8f0, (float*)0x55b068a4bd00);
}
static void InferLayerRelu_23(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_26(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_26(void* img_in, void* img_out) {
  MyConv2dInfer_26(img_in, img_out, (float*)0x7f1accb40010);
}
static void MyBatchNormInfer_26(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_26(void* in_data, void* out_data) {
  MyBatchNormInfer_26(in_data, out_data, (float*)0x55b068a4e970, (float*)0x55b068a4f980,
                      (float*)0x55b068a4c950, (float*)0x55b068a4d960);
}
static void MyConv2dInfer_27(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_27(void* img_in, void* img_out) {
  MyConv2dInfer_27(img_in, img_out, (float*)0x7f1acc93f010);
}
static void MyBatchNormInfer_27(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_27(void* in_data, void* out_data) {
  MyBatchNormInfer_27(in_data, out_data, (float*)0x55b068a529f0, (float*)0x55b068a53a00,
                      (float*)0x55b068a509d0, (float*)0x55b068a519e0);
}
static void InferLayerAdd_7(float* l, float* r, float* out) {
  const auto& len = 200704;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_24(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_7(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_24(in_data, out_data);
  InferLayerBatchNorm_24(out_data, temp_data);
  InferLayerRelu_22(temp_data);
  InferLayerConv2d_25(temp_data, out_data);
  InferLayerBatchNorm_25(out_data, temp_data);
  InferLayerRelu_23(temp_data);
  InferLayerConv2d_26(temp_data, out_data);
  InferLayerBatchNorm_26(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerConv2d_27(in_data, out_data);
  InferLayerBatchNorm_27(out_data, in_data);
  auto short_cut_out = in_data;
  InferLayerAdd_7((float*)bn_out, (float*)short_cut_out, (float*)out_data);
  InferLayerRelu_24(temp_data);
}
static void MyConv2dInfer_28(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_28(void* img_in, void* img_out) {
  MyConv2dInfer_28(img_in, img_out, (float*)0x7f1acc83e010);
}
static void MyBatchNormInfer_28(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_28(void* in_data, void* out_data) {
  MyBatchNormInfer_28(in_data, out_data, (float*)0x55b068a55230, (float*)0x55b068a55640,
                      (float*)0x55b068a54a10, (float*)0x55b068a54e20);
}
static void InferLayerRelu_25(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_29(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_29(void* img_in, void* img_out) {
  MyConv2dInfer_29(img_in, img_out, (float*)0x7f1acc5fd010);
}
static void MyBatchNormInfer_29(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_29(void* in_data, void* out_data) {
  MyBatchNormInfer_29(in_data, out_data, (float*)0x55b068a56290, (float*)0x55b068a566a0,
                      (float*)0x55b068a55a70, (float*)0x55b068a55e80);
}
static void InferLayerRelu_26(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_30(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_30(void* img_in, void* img_out) {
  MyConv2dInfer_30(img_in, img_out, (float*)0x7f1acc4fc010);
}
static void MyBatchNormInfer_30(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_30(void* in_data, void* out_data) {
  MyBatchNormInfer_30(in_data, out_data, (float*)0x55b068a58af0, (float*)0x55b068a59b00,
                      (float*)0x55b068a56ad0, (float*)0x55b068a57ae0);
}
static void InferLayerAdd_8(float* l, float* r, float* out) {
  const auto& len = 200704;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_27(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_8(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_28(in_data, out_data);
  InferLayerBatchNorm_28(out_data, temp_data);
  InferLayerRelu_25(temp_data);
  InferLayerConv2d_29(temp_data, out_data);
  InferLayerBatchNorm_29(out_data, temp_data);
  InferLayerRelu_26(temp_data);
  InferLayerConv2d_30(temp_data, out_data);
  InferLayerBatchNorm_30(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_8((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_27(out_data);
}
static void MyConv2dInfer_31(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_31(void* img_in, void* img_out) {
  MyConv2dInfer_31(img_in, img_out, (float*)0x7f1acc3fb010);
}
static void MyBatchNormInfer_31(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_31(void* in_data, void* out_data) {
  MyBatchNormInfer_31(in_data, out_data, (float*)0x55b068a5b350, (float*)0x55b068a5b760,
                      (float*)0x55b068a5ab30, (float*)0x55b068a5af40);
}
static void InferLayerRelu_28(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_32(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_32(void* img_in, void* img_out) {
  MyConv2dInfer_32(img_in, img_out, (float*)0x7f1acc1ba010);
}
static void MyBatchNormInfer_32(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_32(void* in_data, void* out_data) {
  MyBatchNormInfer_32(in_data, out_data, (float*)0x55b068a5c3b0, (float*)0x55b068a5c7c0,
                      (float*)0x55b068a5bb90, (float*)0x55b068a5bfa0);
}
static void InferLayerRelu_29(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_33(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_33(void* img_in, void* img_out) {
  MyConv2dInfer_33(img_in, img_out, (float*)0x7f1acc0b9010);
}
static void MyBatchNormInfer_33(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_33(void* in_data, void* out_data) {
  MyBatchNormInfer_33(in_data, out_data, (float*)0x55b068a5ec10, (float*)0x55b068a5fc20,
                      (float*)0x55b068a5cbf0, (float*)0x55b068a5dc00);
}
static void InferLayerAdd_9(float* l, float* r, float* out) {
  const auto& len = 200704;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_30(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_9(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_31(in_data, out_data);
  InferLayerBatchNorm_31(out_data, temp_data);
  InferLayerRelu_28(temp_data);
  InferLayerConv2d_32(temp_data, out_data);
  InferLayerBatchNorm_32(out_data, temp_data);
  InferLayerRelu_29(temp_data);
  InferLayerConv2d_33(temp_data, out_data);
  InferLayerBatchNorm_33(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_9((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_30(out_data);
}
static void MyConv2dInfer_34(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_34(void* img_in, void* img_out) {
  MyConv2dInfer_34(img_in, img_out, (float*)0x7f1acbfb8010);
}
static void MyBatchNormInfer_34(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_34(void* in_data, void* out_data) {
  MyBatchNormInfer_34(in_data, out_data, (float*)0x55b068a61470, (float*)0x55b068a61880,
                      (float*)0x55b068a60c50, (float*)0x55b068a61060);
}
static void InferLayerRelu_31(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_35(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_35(void* img_in, void* img_out) {
  MyConv2dInfer_35(img_in, img_out, (float*)0x7f1acbd77010);
}
static void MyBatchNormInfer_35(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_35(void* in_data, void* out_data) {
  MyBatchNormInfer_35(in_data, out_data, (float*)0x55b068a624d0, (float*)0x55b068a628e0,
                      (float*)0x55b068a61cb0, (float*)0x55b068a620c0);
}
static void InferLayerRelu_32(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_36(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_36(void* img_in, void* img_out) {
  MyConv2dInfer_36(img_in, img_out, (float*)0x7f1acbc76010);
}
static void MyBatchNormInfer_36(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_36(void* in_data, void* out_data) {
  MyBatchNormInfer_36(in_data, out_data, (float*)0x55b068a64d30, (float*)0x55b068a65d40,
                      (float*)0x55b068a62d10, (float*)0x55b068a63d20);
}
static void InferLayerAdd_10(float* l, float* r, float* out) {
  const auto& len = 200704;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_33(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_10(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_34(in_data, out_data);
  InferLayerBatchNorm_34(out_data, temp_data);
  InferLayerRelu_31(temp_data);
  InferLayerConv2d_35(temp_data, out_data);
  InferLayerBatchNorm_35(out_data, temp_data);
  InferLayerRelu_32(temp_data);
  InferLayerConv2d_36(temp_data, out_data);
  InferLayerBatchNorm_36(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_10((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_33(out_data);
}
static void MyConv2dInfer_37(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_37(void* img_in, void* img_out) {
  MyConv2dInfer_37(img_in, img_out, (float*)0x7f1acbb75010);
}
static void MyBatchNormInfer_37(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_37(void* in_data, void* out_data) {
  MyBatchNormInfer_37(in_data, out_data, (float*)0x55b068a67590, (float*)0x55b068a679a0,
                      (float*)0x55b068a66d70, (float*)0x55b068a67180);
}
static void InferLayerRelu_34(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_38(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_38(void* img_in, void* img_out) {
  MyConv2dInfer_38(img_in, img_out, (float*)0x7f1acb934010);
}
static void MyBatchNormInfer_38(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_38(void* in_data, void* out_data) {
  MyBatchNormInfer_38(in_data, out_data, (float*)0x55b068a685f0, (float*)0x55b068a68a00,
                      (float*)0x55b068a67dd0, (float*)0x55b068a681e0);
}
static void InferLayerRelu_35(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_39(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_39(void* img_in, void* img_out) {
  MyConv2dInfer_39(img_in, img_out, (float*)0x7f1acb833010);
}
static void MyBatchNormInfer_39(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_39(void* in_data, void* out_data) {
  MyBatchNormInfer_39(in_data, out_data, (float*)0x55b068a6ae50, (float*)0x55b068a6be60,
                      (float*)0x55b068a68e30, (float*)0x55b068a69e40);
}
static void InferLayerAdd_11(float* l, float* r, float* out) {
  const auto& len = 200704;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_36(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_11(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_37(in_data, out_data);
  InferLayerBatchNorm_37(out_data, temp_data);
  InferLayerRelu_34(temp_data);
  InferLayerConv2d_38(temp_data, out_data);
  InferLayerBatchNorm_38(out_data, temp_data);
  InferLayerRelu_35(temp_data);
  InferLayerConv2d_39(temp_data, out_data);
  InferLayerBatchNorm_39(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_11((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_36(out_data);
}
static void MyConv2dInfer_40(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_40(void* img_in, void* img_out) {
  MyConv2dInfer_40(img_in, img_out, (float*)0x7f1acb732010);
}
static void MyBatchNormInfer_40(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_40(void* in_data, void* out_data) {
  MyBatchNormInfer_40(in_data, out_data, (float*)0x55b068a6d6b0, (float*)0x55b068a6dac0,
                      (float*)0x55b068a6ce90, (float*)0x55b068a6d2a0);
}
static void InferLayerRelu_37(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_41(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_41(void* img_in, void* img_out) {
  MyConv2dInfer_41(img_in, img_out, (float*)0x7f1acb4f1010);
}
static void MyBatchNormInfer_41(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_41(void* in_data, void* out_data) {
  MyBatchNormInfer_41(in_data, out_data, (float*)0x55b068a6e710, (float*)0x55b068a6eb20,
                      (float*)0x55b068a6def0, (float*)0x55b068a6e300);
}
static void InferLayerRelu_38(void* img_in) {
  const auto& len = 50176;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_42(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_42(void* img_in, void* img_out) {
  MyConv2dInfer_42(img_in, img_out, (float*)0x7f1acb3f0010);
}
static void MyBatchNormInfer_42(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_42(void* in_data, void* out_data) {
  MyBatchNormInfer_42(in_data, out_data, (float*)0x55b068a70f70, (float*)0x55b068a71f80,
                      (float*)0x55b068a6ef50, (float*)0x55b068a6ff60);
}
static void InferLayerAdd_12(float* l, float* r, float* out) {
  const auto& len = 200704;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_39(void* img_in) {
  const auto& len = 200704;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_12(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_40(in_data, out_data);
  InferLayerBatchNorm_40(out_data, temp_data);
  InferLayerRelu_37(temp_data);
  InferLayerConv2d_41(temp_data, out_data);
  InferLayerBatchNorm_41(out_data, temp_data);
  InferLayerRelu_38(temp_data);
  InferLayerConv2d_42(temp_data, out_data);
  InferLayerBatchNorm_42(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_12((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_39(out_data);
}
static void MyConv2dInfer_43(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_43(void* img_in, void* img_out) {
  MyConv2dInfer_43(img_in, img_out, (float*)0x7f1acb1ef010);
}
static void MyBatchNormInfer_43(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_43(void* in_data, void* out_data) {
  MyBatchNormInfer_43(in_data, out_data, (float*)0x55b068a73fd0, (float*)0x55b068a747e0,
                      (float*)0x55b068a72fb0, (float*)0x55b068a737c0);
}
static void InferLayerRelu_40(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_44(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_44(void* img_in, void* img_out) {
  MyConv2dInfer_44(img_in, img_out, (float*)0x7f1aca8ee010);
}
static void MyBatchNormInfer_44(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_44(void* in_data, void* out_data) {
  MyBatchNormInfer_44(in_data, out_data, (float*)0x55b068a76030, (float*)0x55b068a76840,
                      (float*)0x55b068a75010, (float*)0x55b068a75820);
}
static void InferLayerRelu_41(void* img_in) {
  const auto& len = 25088;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_45(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_45(void* img_in, void* img_out) {
  MyConv2dInfer_45(img_in, img_out, (float*)0x7f1aca4ed010);
}
static void MyBatchNormInfer_45(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_45(void* in_data, void* out_data) {
  MyBatchNormInfer_45(in_data, out_data, (float*)0x55b068a7b090, (float*)0x55b068a7d0a0,
                      (float*)0x55b068a77070, (float*)0x55b068a79080);
}
static void MyConv2dInfer_46(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_46(void* img_in, void* img_out) {
  MyConv2dInfer_46(img_in, img_out, (float*)0x7f1ac9cec010);
}
static void MyBatchNormInfer_46(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_46(void* in_data, void* out_data) {
  MyBatchNormInfer_46(in_data, out_data, (float*)0x55b068a83110, (float*)0x55b068a85120,
                      (float*)0x55b068a7f0f0, (float*)0x55b068a81100);
}
static void InferLayerAdd_13(float* l, float* r, float* out) {
  const auto& len = 100352;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_42(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_13(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_43(in_data, out_data);
  InferLayerBatchNorm_43(out_data, temp_data);
  InferLayerRelu_40(temp_data);
  InferLayerConv2d_44(temp_data, out_data);
  InferLayerBatchNorm_44(out_data, temp_data);
  InferLayerRelu_41(temp_data);
  InferLayerConv2d_45(temp_data, out_data);
  InferLayerBatchNorm_45(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerConv2d_46(in_data, out_data);
  InferLayerBatchNorm_46(out_data, in_data);
  auto short_cut_out = in_data;
  InferLayerAdd_13((float*)bn_out, (float*)short_cut_out, (float*)out_data);
  InferLayerRelu_42(temp_data);
}
static void MyConv2dInfer_47(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_47(void* img_in, void* img_out) {
  MyConv2dInfer_47(img_in, img_out, (float*)0x7f1ac98eb010);
}
static void MyBatchNormInfer_47(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_47(void* in_data, void* out_data) {
  MyBatchNormInfer_47(in_data, out_data, (float*)0x55b068a88150, (float*)0x55b068a88960,
                      (float*)0x55b068a87130, (float*)0x55b068a87940);
}
static void InferLayerRelu_43(void* img_in) {
  const auto& len = 25088;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_48(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_48(void* img_in, void* img_out) {
  MyConv2dInfer_48(img_in, img_out, (float*)0x7f1ac8fea010);
}
static void MyBatchNormInfer_48(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_48(void* in_data, void* out_data) {
  MyBatchNormInfer_48(in_data, out_data, (float*)0x55b068a8a1b0, (float*)0x55b068a8a9c0,
                      (float*)0x55b068a89190, (float*)0x55b068a899a0);
}
static void InferLayerRelu_44(void* img_in) {
  const auto& len = 25088;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_49(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_49(void* img_in, void* img_out) {
  MyConv2dInfer_49(img_in, img_out, (float*)0x7f1ac8be9010);
}
static void MyBatchNormInfer_49(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_49(void* in_data, void* out_data) {
  MyBatchNormInfer_49(in_data, out_data, (float*)0x55b068a8f210, (float*)0x55b068a91220,
                      (float*)0x55b068a8b1f0, (float*)0x55b068a8d200);
}
static void InferLayerAdd_14(float* l, float* r, float* out) {
  const auto& len = 100352;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_45(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_14(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_47(in_data, out_data);
  InferLayerBatchNorm_47(out_data, temp_data);
  InferLayerRelu_43(temp_data);
  InferLayerConv2d_48(temp_data, out_data);
  InferLayerBatchNorm_48(out_data, temp_data);
  InferLayerRelu_44(temp_data);
  InferLayerConv2d_49(temp_data, out_data);
  InferLayerBatchNorm_49(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_14((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_45(out_data);
}
static void MyConv2dInfer_50(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_50(void* img_in, void* img_out) {
  MyConv2dInfer_50(img_in, img_out, (float*)0x7f1ac87e8010);
}
static void MyBatchNormInfer_50(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_50(void* in_data, void* out_data) {
  MyBatchNormInfer_50(in_data, out_data, (float*)0x55b068a94270, (float*)0x55b068a94a80,
                      (float*)0x55b068a93250, (float*)0x55b068a93a60);
}
static void InferLayerRelu_46(void* img_in) {
  const auto& len = 25088;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_51(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_51(void* img_in, void* img_out) {
  MyConv2dInfer_51(img_in, img_out, (float*)0x7f1ac7ee7010);
}
static void MyBatchNormInfer_51(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_51(void* in_data, void* out_data) {
  MyBatchNormInfer_51(in_data, out_data, (float*)0x55b068a962d0, (float*)0x55b068a96ae0,
                      (float*)0x55b068a952b0, (float*)0x55b068a95ac0);
}
static void InferLayerRelu_47(void* img_in) {
  const auto& len = 25088;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void MyConv2dInfer_52(void* img_in, void* img_out, float* weight) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerConv2d_52(void* img_in, void* img_out) {
  MyConv2dInfer_52(img_in, img_out, (float*)0x7f1ac7ae6010);
}
static void MyBatchNormInfer_52(
    void* img_in, void* img_out, float* mean, float* var, float* gamma, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerBatchNorm_52(void* in_data, void* out_data) {
  MyBatchNormInfer_52(in_data, out_data, (float*)0x55b068a9b330, (float*)0x55b068a9d340,
                      (float*)0x55b068a97310, (float*)0x55b068a99320);
}
static void InferLayerAdd_15(float* l, float* r, float* out) {
  const auto& len = 100352;
  const int vec_size = 8;
  __m256 l_vec, r_vec, res_vec;
  for (int i = 0; i < len; i += vec_size) {
    l_vec = _mm256_loadu_ps(l + i);
    r_vec = _mm256_loadu_ps(r + i);
    res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(out + i, res_vec);
  }
}
static void InferLayerRelu_48(void* img_in) {
  const auto& len = 100352;
  float* img = (float*)img_in;
  for (int i = 0; i < len; ++i) {
    img[i] = img[i] > 0 ? img[i] : 0;
  }
}
static void InferBottleNeck_15(void* in_data, void* out_data, void* temp_data, bool down_sample) {
  InferLayerConv2d_50(in_data, out_data);
  InferLayerBatchNorm_50(out_data, temp_data);
  InferLayerRelu_46(temp_data);
  InferLayerConv2d_51(temp_data, out_data);
  InferLayerBatchNorm_51(out_data, temp_data);
  InferLayerRelu_47(temp_data);
  InferLayerConv2d_52(temp_data, out_data);
  InferLayerBatchNorm_52(out_data, temp_data);
  auto bn_out = temp_data;
  InferLayerAdd_15((float*)bn_out, (float*)in_data, (float*)out_data);
  InferLayerRelu_48(out_data);
}
static void MyAvgPoolInfer(void* img_in, void* img_out) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
}
static void InferLayerAvgPool(void* in_data, void* out_data) { MyAvgPoolInfer(in_data, out_data); };
static void MyFullyConnectInfer(void* img_in, void* img_out, float* weight, float* bias) {
  float* img = (float*)img_in;
  float* out = (float*)img_out;
  for (int outer = 0; outer < 1000; ++outer) {
    float sum_x = float(0);
    const int vec_size = 8;
    __m256 l_vec, weight_vec;
    for (int inner = 0; inner < 2048; inner += vec_size) {
      l_vec = _mm256_loadu_ps(&img[inner]);
      weight_vec = _mm256_loadu_ps(&weight[outer * 2048 + inner]);
      l_vec = _mm256_mul_ps(l_vec, weight_vec);
      sum_x += avx2_sum(l_vec);
    }
    out[outer] = sum_x + bias[outer];
  }
}
static void InferLayerFC(void* img_in, void* img_out) {
  MyFullyConnectInfer(img_in, img_out, (float*)0x7f1ac7315010, (float*)0x55b068a9f350);
}
void InferCodeGen(void* mem_main0, void* mem_main1, void* mem_temp) {
  InferLayerConv2d_0(mem_main0, mem_main1);
  InferLayerBatchNorm_0(mem_main1, mem_main0);
  InferLayerRelu_0(mem_main0);
  InferLayerMaxPool(mem_main0, mem_main1);
  // layer1
  InferBottleNeck_0(mem_main1, mem_main0, mem_temp, true);
  InferBottleNeck_1(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck_2(mem_main1, mem_main0, mem_temp, false);
  // layer2
  InferBottleNeck_3(mem_main0, mem_main1, mem_temp, true);
  InferBottleNeck_4(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck_5(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck_6(mem_main1, mem_main0, mem_temp, false);
  // layer3
  InferBottleNeck_7(mem_main0, mem_main1, mem_temp, true);
  InferBottleNeck_8(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck_9(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck_10(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck_11(mem_main0, mem_main1, mem_temp, false);
  InferBottleNeck_12(mem_main1, mem_main0, mem_temp, false);
  // layer4
  InferBottleNeck_13(mem_main0, mem_main1, mem_temp, true);
  InferBottleNeck_14(mem_main1, mem_main0, mem_temp, false);
  InferBottleNeck_15(mem_main0, mem_main1, mem_temp, false);
  // avg pool
  InferLayerAvgPool(mem_main1, mem_main0);
  // Linear
  InferLayerFC(mem_main0, mem_main1);
}
