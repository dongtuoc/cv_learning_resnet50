#include "resnet_codegen.h"

#include <dlfcn.h>
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "ops/bn.h"
#include "ops/common.h"
#include "ops/conv2d.h"
#include "ops/fc.h"
#include "ops/pool.h"

extern int conv_idx;
extern int bn_cnt;
extern int relu_cnt;
extern int add_cnt;
extern int bottle_idx;

// optimize by pre-load params of networks
void* __global_weight[MAX_MEM_NUM] = {nullptr};

int put_cnt = 0;
int out_cnt = 0;

template <typename T>
void* LoadData(const std::string& file_name, int len, bool is_float) {
  T* data = (T*)malloc(len * sizeof(T));
  FILE* fp = fopen(file_name.c_str(), "r");
  // std::cout << "file_name = " << file_name << ", fp = " << fp << std::endl;
  for (auto i = 0; i < len; i++) {
    float x = 0;
    fscanf(fp, "%f", &x);
    data[i] = is_float ? x : (int)x;
  }
  fclose(fp);
  __global_weight[put_cnt++] = data;
  return (void*)data;
}

float* LoadCon2dWeightPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_weight.txt";
  return (float*)LoadData<float>(file_name, len, true);
}

float* LoadCon2dWeight() { return (float*)__global_weight[out_cnt++]; }

int* LoadCon2dParamPreLoad(const std::string& name, int len) {
  auto file_name = "../../model/resnet50_weight/resnet50_" + name + "_param.txt";
  return (int*)LoadData<int>(file_name, len, false);
}

int* LoadCon2dParam() { return (int*)__global_weight[out_cnt++]; }

void ComputeLayerReluPreLoad(void* img_in, int len) { return; }

int relu_cnt = 0;
void ComputeLayerRelu(void* img_in, int len) {
  std::ostringstream relu_os;
  relu_os << "inline void ComputeLayerRelu_" << relu_cnt << "(float* img) {\n";
  relu_os << "  __m256 zero = _mm256_setzero_ps();\n";
  relu_os << "  __m256 in_vec;\n";
  relu_os << "  const int vec_size = 8;\n";
  relu_os << "  for (int l = 0; l < " << len << "; l += vec_size) {\n";
  relu_os << "    in_vec = _mm256_loadu_ps(&img[l]);\n";
  relu_os << "    _mm256_storeu_ps(&img[l], _mm256_max_ps(in_vec, zero));\n";
  relu_os << "  }\n";
  relu_os << "}\n";
  write("codegen/relu.h", relu_os);
  relu_cnt++;
  return;
}

void ComputeLayerConv2dPreLoad(void* img_in,
                               void* img_out,
                               int hi,
                               int wi,
                               int& ho,
                               int& wo,
                               int& co,
                               const std::string& layer_name) {
  auto param = LoadCon2dParamPreLoad(layer_name, 5);
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeightPreLoad(layer_name, co * kernel * kernel * ci);
  if (hi == 224) {
    return MyConv2dPreLoad(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad);
  } else {
    return MyConv2dPreLoad(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad);
  }
}

void ComputeLayerConv2d(void* img_in, void* img_out, int hi, int wi, int& ho, int& wo, int& co) {
  auto param = LoadCon2dParam();
  // ci, co, kernel, stride, pad
  auto ci = param[0];
  co = param[1];
  auto kernel = param[2];
  auto stride = param[3];
  auto pad = param[4];
  auto weight = LoadCon2dWeight();
  if (hi == 224) {
    MyConv2d(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, true);
  } else {
    MyConv2d(img_in, img_out, weight, hi, wi, ho, wo, ci, co, kernel, stride, pad, false);
  }

  std::ostringstream dec_os;
  dec_os << "inline void ComputeLayerConv2d_" << conv_idx << "(void* img_in, void* img_out) {\n";
  dec_os << "  MyConv2d_" << conv_idx << "(img_in, img_out, (float *)" << weight << ");\n";
  dec_os << "}\n";
  const std::string fd = "codegen/conv_" + std::to_string(conv_idx) + ".h";
  write(fd, dec_os);
  conv_idx++;
  return;
}

void ComputeLayerFCPreLoad(void* img_in, void* img_out, const std::string& layer_name) {
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  LoadData<float>(weight_file_name, 1000 * 2048, true);
  LoadData<float>(bias_file_name, 1000, true);
  return;
}

void ComputeLayerFC(void* img_in, void* img_out) {
  auto weight = (float*)__global_weight[out_cnt++];
  auto bias = (float*)__global_weight[out_cnt++];
  MyFC(img_in, img_out, weight, bias);

  std::ostringstream imp_os;
  std::ostringstream dec_os;
  dec_os << "void ComputeLayerFC(void* img_in, void* img_out);\n";
  imp_os << "void ComputeLayerFC(void* img_in, void* img_out) {\n";
  imp_os << "  MyFC(img_in, img_out, (float *)" << weight << ", (float *)" << bias << ");\n";
  imp_os << "}\n";
  const std::string fi = "codegen/fc.cc";
  const std::string fd = "codegen/fc.h";
  write(fi, imp_os);
  write(fd, dec_os);
  return;
}

void ComputeLayerBatchNormPreLoad(
    void* in_data, void* out_data, int h, int w, int c, const std::string& layer_name) {
  auto weight_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_weight.txt";
  auto bias_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_bias.txt";
  auto mean_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_mean.txt";
  auto var_file_name = "../../model/resnet50_weight/resnet50_" + layer_name + "_running_var.txt";
  LoadData<float>(weight_file_name, c, true);
  LoadData<float>(bias_file_name, c, true);
  LoadData<float>(mean_file_name, c, true);
  LoadData<float>(var_file_name, c, true);
  return;
}

void ComputeLayerBatchNorm(void* in_data, void* out_data, int h, int w, int c) {
  auto gamma = (float*)__global_weight[out_cnt++];
  auto bias = (float*)__global_weight[out_cnt++];
  auto mean = (float*)__global_weight[out_cnt++];
  auto var = (float*)__global_weight[out_cnt++];
  MyBatchNorm(in_data, out_data, mean, var, gamma, bias, h, w, c);

  std::ostringstream dec_os;
  dec_os << "inline void ComputeLayerBatchNorm_" << bn_cnt << "(void* in_data, void* out_data) {\n";
  dec_os << "  MyBatchNorm_" << bn_cnt << "(in_data, out_data, (float*)" << mean << ", (float*)"
         << var << ", (float*)" << gamma << ", (float *)" << bias << ");\n";
  dec_os << "}\n";
  const std::string fd = "codegen/bn.h";
  write(fd, dec_os);
  bn_cnt++;
  return;
}

void ComputeLayerMaxPoolPreLoad(void* in_data, void* out_data) { return; }

void ComputeLayerMaxPool(void* in_data, void* out_data) {
  MyMaxPool(in_data, out_data);

  std::ostringstream dec_os;
  dec_os << "inline void ComputeLayerMaxPool(void* in_data, void* out_data) {\n";
  dec_os << "  MyMaxPool(in_data, out_data);\n";
  dec_os << "}\n";
  const std::string fd = "codegen/maxpool.h";
  write(fd, dec_os);
}

void ComputeLayerAvgPoolPreLoad(void* in_data, void* out_data) { return; }

void ComputeLayerAvgPool(void* in_data, void* out_data) {
  MyAvgPool(in_data, out_data);

  std::ostringstream dec_os;
  dec_os << "inline void ComputeLayerAvgPool(void* in_data, void* out_data) {\n";
  dec_os << "  MyAvgPool(in_data, out_data);\n";
  dec_os << "}\n";
  const std::string fd = "codegen/avgpool.h";
  write(fd, dec_os);
}

void AddPreLoad(float* l, float* r, float* out, int len) { return; }

int add_cnt = 0;
void Add(float* l, float* r, float* out, int len) {
  std::ostringstream dec_os;
  if (add_cnt == 0) {
    dec_os << "#include <immintrin.h>\n";
  }
  dec_os << "inline void Add_" << add_cnt << "(float* l, float* r, float* out) {\n";
  dec_os << "  const int vec_size = 8;\n";
  dec_os << "  __m256 l_vec, r_vec, res_vec;\n";
  dec_os << "  for (int i = 0; i < " << len << "; i += vec_size) {\n";
  dec_os << "    l_vec = _mm256_loadu_ps(l + i);\n";
  dec_os << "    r_vec = _mm256_loadu_ps(r + i);\n";
  dec_os << "    res_vec = _mm256_add_ps(l_vec, r_vec);\n";
  dec_os << "    _mm256_storeu_ps(out + i, res_vec);\n";
  dec_os << "  }\n";
  dec_os << "}\n";
  const std::string fd = "codegen/add.h";
  write(fd, dec_os);
  add_cnt++;
  return;
}

void ComputeBottleNeckPreLoad(void* in_data,
                              void* out_data,
                              void* temp_data,
                              int hi,
                              int wi,
                              int& ho,
                              int& wo,
                              int& co,
                              const std::string& bottleneck_layer_name,
                              bool down_sample) {
  int h0, w0, c0;
  int h1, w1, c1;

  ComputeLayerConv2dPreLoad(in_data, out_data, hi, wi, h0, w0, c0,
                            bottleneck_layer_name + "_conv1");
  ComputeLayerBatchNormPreLoad(out_data, temp_data, h0, w0, c0,
                               bottleneck_layer_name + std::string("_bn1"));
  ComputeLayerReluPreLoad(temp_data, h0 * w0 * c0);

  ComputeLayerConv2dPreLoad(temp_data, out_data, h0, w0, h1, w1, c1,
                            bottleneck_layer_name + std::string("_conv2"));
  ComputeLayerBatchNormPreLoad(out_data, temp_data, h1, w1, c1,
                               bottleneck_layer_name + std::string("_bn2"));
  ComputeLayerReluPreLoad(temp_data, h1 * w1 * c1);

  ComputeLayerConv2dPreLoad(temp_data, out_data, h1, w1, h0, w0, c0,
                            bottleneck_layer_name + std::string("_conv3"));
  ComputeLayerBatchNormPreLoad(out_data, temp_data, h0, w0, c0,
                               bottleneck_layer_name + std::string("_bn3"));
  auto bn_out = temp_data;

  if (down_sample) {
    int h2, w2, c2;
    ComputeLayerConv2dPreLoad(in_data, out_data, hi, wi, h2, w2, c2,
                              bottleneck_layer_name + std::string("_downsample_conv2d"));
    ComputeLayerBatchNormPreLoad(out_data, in_data, h2, w2, c2,
                                 bottleneck_layer_name + std::string("_downsample_batchnorm"));
    auto short_cut_out = in_data;
    AddPreLoad((float*)bn_out, (float*)short_cut_out, (float*)out_data, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    return ComputeLayerReluPreLoad(out_data, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    AddPreLoad((float*)bn_out, (float*)in_data, (float*)out_data, h0 * w0 * c0);
    return ComputeLayerReluPreLoad(out_data, h0 * w0 * c0);
  }
}

int bottle_idx = 0;
void ComputeBottleNeck(void* in_data,
                       void* out_data,
                       void* temp_data,
                       int hi,
                       int wi,
                       int& ho,
                       int& wo,
                       int& co,
                       bool down_sample) {
  int h0, w0, c0;
  int h1, w1, c1;
  auto this_conv_idx = conv_idx;
  auto this_bn_idx = bn_cnt;
  auto this_relu_idx = relu_cnt;
  auto this_add_idx = add_cnt;

  ComputeLayerConv2d(in_data, out_data, hi, wi, h0, w0, c0);
  ComputeLayerBatchNorm(out_data, temp_data, h0, w0, c0);
  ComputeLayerRelu(temp_data, h0 * w0 * c0);

  ComputeLayerConv2d(temp_data, out_data, h0, w0, h1, w1, c1);
  ComputeLayerBatchNorm(out_data, temp_data, h1, w1, c1);
  ComputeLayerRelu(temp_data, h1 * w1 * c1);

  ComputeLayerConv2d(temp_data, out_data, h1, w1, h0, w0, c0);
  ComputeLayerBatchNorm(out_data, temp_data, h0, w0, c0);
  auto bn_out = temp_data;

  if (down_sample) {
    int h2, w2, c2;
    ComputeLayerConv2d(in_data, out_data, hi, wi, h2, w2, c2);
    ComputeLayerBatchNorm(out_data, in_data, h2, w2, c2);
    auto short_cut_out = in_data;
    Add((float*)bn_out, (float*)short_cut_out, (float*)out_data, h2 * w2 * c2);
    ho = h2, wo = w2, co = c2;
    ComputeLayerRelu(out_data, h2 * w2 * c2);
  } else {
    ho = h0, wo = w0, co = c0;
    Add((float*)bn_out, (float*)in_data, (float*)out_data, h0 * w0 * c0);
    ComputeLayerRelu(out_data, h0 * w0 * c0);
  }

  std::ostringstream bottle_os;
  bottle_os << "void ComputeBottleNeck_" << bottle_idx
            << "(void* in_data, void* out_data, void* temp_data) {\n";
  bottle_os << "  ComputeLayerConv2d_" << this_conv_idx++ << "(in_data, out_data);\n";
  bottle_os << "  ComputeLayerBatchNorm_" << this_bn_idx++ << "(out_data, temp_data);\n";
  bottle_os << "  ComputeLayerRelu_" << this_relu_idx++ << "((float*)temp_data);\n";

  bottle_os << "  ComputeLayerConv2d_" << this_conv_idx++ << "(temp_data, out_data);\n";
  bottle_os << "  ComputeLayerBatchNorm_" << this_bn_idx++ << "(out_data, temp_data);\n";
  bottle_os << "  ComputeLayerRelu_" << this_relu_idx++ << "((float*)temp_data);\n";

  bottle_os << "  ComputeLayerConv2d_" << this_conv_idx++ << "(temp_data, out_data);\n";
  bottle_os << "  ComputeLayerBatchNorm_" << this_bn_idx++ << "(out_data, temp_data);\n";
  bottle_os << "  auto bn_out = temp_data;\n";
  if (down_sample) {
    bottle_os << "  ComputeLayerConv2d_" << this_conv_idx++ << "(in_data, out_data);\n";
    bottle_os << "  ComputeLayerBatchNorm_" << this_bn_idx++ << "(out_data, in_data);\n";
    bottle_os << "  auto short_cut_out = in_data;\n";
    bottle_os << "  Add_" << this_add_idx++
              << "((float*)bn_out, (float*)short_cut_out, (float*)out_data);\n";
    bottle_os << "  ComputeLayerRelu_" << this_relu_idx++ << "((float *)out_data);\n";
  } else {
    bottle_os << "  Add_" << this_add_idx++
              << "((float*)bn_out, (float*)in_data, (float*)out_data);\n";
    bottle_os << "  ComputeLayerRelu_" << this_relu_idx++ << "((float *)out_data);\n";
  }
  bottle_os << "}\n";
  write("codegen/bottle.h", bottle_os);
  bottle_idx++;
  return;
}

void PreLoadParams() {
  float* img0 = nullptr;
  float* img1 = nullptr;
  float* img2 = nullptr;
  int h0, w0, c0;
  int h1, w1, c1;
  ComputeLayerConv2dPreLoad(img0, img1, 224, 224, h1, w1, c1, "conv1");
  ComputeLayerBatchNormPreLoad(img1, img0, h1, w1, c1, "bn1");
  ComputeLayerReluPreLoad(img0, h1 * w1 * c1);
  ComputeLayerMaxPoolPreLoad(img0, img1);
  // layer1
  ComputeBottleNeckPreLoad(img1, img0, img2, 56, 56, h1, w1, c1, "layer1_bottleneck0", true);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer1_bottleneck1", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer1_bottleneck2", false);
  // layer2
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck0", true);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck1", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer2_bottleneck2", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer2_bottleneck3", false);
  // layer3
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck0", true);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck1", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck2", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck3", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer3_bottleneck4", false);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer3_bottleneck5", false);
  // layer4
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck0", true);
  ComputeBottleNeckPreLoad(img1, img0, img2, h0, w0, h1, w1, c1, "layer4_bottleneck1", false);
  ComputeBottleNeckPreLoad(img0, img1, img2, h1, w1, h0, w0, c0, "layer4_bottleneck2", false);
  // avg pool
  ComputeLayerAvgPoolPreLoad(img1, img0);
  // Linear
  ComputeLayerFCPreLoad(img0, img1, "fc");
  return;
}

void CodeGen(void* mem_main0, void* mem_main1, void* mem_temp) {
  std::cout << "\033[0;32mCode Gen Start \033[0m" << std::endl;
  int res = std::system("rm codegen -rf; mkdir codegen");

  std::ostringstream header_os;
  header_os << "#include <immintrin.h>\n";
  header_os << "#include <cmath>\n";
  header_os << "#include <cstdint>\n";
  header_os << "#include <iostream>\n";
  header_os << "#include <opencv2/opencv.hpp>\n";
  header_os << "#include <string>\n";
  header_os << "#include <vector>\n";
  header_os << "#include \"maxpool.h\"\n";
  header_os << "#include \"avgpool.h\"\n";
  header_os << "#include \"relu.h\"\n";
  header_os << "#include \"add.h\"\n";
  header_os << "#include \"bn.h\"\n";
  header_os << "#include \"fc.h\"\n";
  for (int i = 0; i < 53; i++) {
    std::string inc = "#include \"conv_" + std::to_string(i) + ".h\"\n";
    header_os << inc;
  }
  header_os << "#include \"bottle.h\"\n";

  std::ostringstream func_os;

  func_os << "#include <immintrin.h>\n";
  func_os << "inline float avx2_sum(__m256 in_vec) {\n";
  func_os << "  in_vec = _mm256_add_ps(in_vec, _mm256_permute2f128_ps(in_vec, in_vec, 1));\n";
  func_os << "  in_vec = _mm256_hadd_ps(in_vec, in_vec);\n";
  func_os << "  return _mm256_cvtss_f32(_mm256_hadd_ps(in_vec, in_vec));\n";
  func_os << "}\n";
  write("codegen/func.h", func_os);

  out_cnt = 0;
  conv_idx = 0;
  bn_cnt = 0;
  relu_cnt = 0;
  add_cnt = 0;
  bottle_idx = 0;
  int h0, w0, c0;
  int h1, w1, c1;

  ComputeLayerConv2d(mem_main0, mem_main1, 224, 224, h1, w1, c1);
  ComputeLayerBatchNorm(mem_main1, mem_main0, h1, w1, c1);
  ComputeLayerRelu(mem_main0, h1 * w1 * c1);
  ComputeLayerMaxPool(mem_main0, mem_main1);
  // layer1
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, 56, 56, h1, w1, c1, true);
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, false);
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1, false);
  // layer2
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, true);
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1, false);
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, false);
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1, false);
  // layer3
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, true);
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1, false);
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, false);
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1, false);
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, false);
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1, false);
  // layer4
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, true);
  ComputeBottleNeck(mem_main1, mem_main0, mem_temp, h0, w0, h1, w1, c1, false);
  ComputeBottleNeck(mem_main0, mem_main1, mem_temp, h1, w1, h0, w0, c0, false);
  // avg pool
  ComputeLayerAvgPool(mem_main1, mem_main0);
  // Linear
  ComputeLayerFC(mem_main0, mem_main1);

  header_os << "void Infer(void* mem_main0, void* mem_main1, void* mem_temp) {\n";
  header_os << "  ComputeLayerConv2d_0(mem_main0, mem_main1);\n";
  header_os << "  ComputeLayerBatchNorm_0(mem_main1, mem_main0);\n";
  header_os << "  ComputeLayerRelu_0((float *)mem_main0);\n";
  header_os << "  ComputeLayerMaxPool(mem_main0, mem_main1);\n";
  header_os << "  // layer1\n";
  header_os << "  ComputeBottleNeck_0(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  ComputeBottleNeck_1(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  ComputeBottleNeck_2(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  // layer2\n";
  header_os << "  ComputeBottleNeck_3(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  ComputeBottleNeck_4(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  ComputeBottleNeck_5(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  ComputeBottleNeck_6(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  // layer3\n";
  header_os << "  ComputeBottleNeck_7(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  ComputeBottleNeck_8(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  ComputeBottleNeck_9(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  ComputeBottleNeck_10(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  ComputeBottleNeck_11(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  ComputeBottleNeck_12(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  // layer4\n";
  header_os << "  ComputeBottleNeck_13(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  ComputeBottleNeck_14(mem_main1, mem_main0, mem_temp);\n";
  header_os << "  ComputeBottleNeck_15(mem_main0, mem_main1, mem_temp);\n";
  header_os << "  // avg pool\n";
  header_os << "  ComputeLayerAvgPool(mem_main1, mem_main0);\n";
  header_os << "  // Linear\n";
  header_os << "  ComputeLayerFC(mem_main0, mem_main1);\n";
  header_os << "}\n";

  write("codegen/codegen.cc", header_os);
  std::cout << "\033[0;32mCode Gen Done \033[0m" << std::endl;
}

void CompileModule() {
  std::cout << "\033[0;32mBegin Compile Module \033[0m" << std::endl;
  std::system("rm -rf codegen/*.so codegen/*.o;");
  std::string cmd_common;
  std::string cmd;
  std::string opt_level = " -Ofast ";
  std::string opencv_install_dir = "/usr/include/opencv4";
  std::string codegen_dir = "codegen";
  cmd_common = "g++ -mavx -fPIC " + opt_level + "-I" + opencv_install_dir +
               " -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs ";

  int compile_step = 1;
  const int total_steps = 59;
  auto CompileStepVision = [&](int& step, const std::string& cmd) {
    float pre = (float)step / (float)total_steps * 100;
    auto pre_label = "[" + std::to_string((int)pre) + "%]";
    std::cout << pre_label << " \033[0;32m" << cmd << "\033[0m" << std::endl;
    step++;
  };

  // compile .cc to .o
  std::vector<std::string> obj_list;
  auto Compile2Obj = [&](const std::string& layer_name) {
    auto cmd = cmd_common + "-c " + codegen_dir + "/" + layer_name + ".cc" + " -o " + codegen_dir +
               "/" + layer_name + ".o";
    CompileStepVision(compile_step, cmd);
    std::system(cmd.c_str());
    obj_list.push_back(codegen_dir + "/" + layer_name + ".o");
  };
  Compile2Obj("maxpool");
  Compile2Obj("avgpool");
  Compile2Obj("bn");
  Compile2Obj("fc");
  Compile2Obj("codegen");
  for (int i = 0; i < 53; i++) {
    Compile2Obj("conv_" + std::to_string(i));
  }

  // link .o to libresnet.so
  cmd = cmd_common + "-shared -o " + codegen_dir + "/libresnet.so ";
  for (auto it : obj_list) {
    cmd += it + " ";
  }
  CompileStepVision(compile_step, cmd);
  std::system(cmd.c_str());
  std::system("rm -f codegen/*.o;");
  std::cout << "\033[0;32mEnd Compile Module \033[0m" << std::endl;
}

Module LoadModule() {
  void* handle = dlopen("./codegen/libresnet.so", RTLD_LAZY);
  if (!handle) {
    printf("Error dlopen.\n");
    return 0;
  }
  auto myFunc = (Module)dlsym(handle, "_Z5InferPvS_S_");
  if (!myFunc) {
    printf("Error with dlsym.\n");
    return 0;
  } else {
    printf("Succ Get myFunc\n");
  }
  return myFunc;
}
