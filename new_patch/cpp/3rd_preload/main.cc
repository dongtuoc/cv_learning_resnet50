#include <dirent.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../label.h"
#include "resnet_preload.h"

extern std::map<std::string, void*> __global_params;

static std::vector<std::string> GetFileName() {
  std::vector<std::string> filenames;
  std::string dir_path("../../pics/ani_12/");
  // filenames.push_back(dir_path + std::string("Niu.jpg"));
  // return filenames;
  DIR* dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    std::cerr << "Failed to open directory: " << dir_path << std::endl;
    exit(0);
  }
  dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG) {
      filenames.push_back(dir_path + std::string(entry->d_name));
    }
  }
  closedir(dir);
  std::cout << "Read Files:" << std::endl;
  for (const auto& filename : filenames) {
    std::cout << filename << std::endl;
  }
  std::cout << "\n\n" << std::endl;
  return filenames;
}

static float* PreProcess(const std::string& file_name) {
  auto transpose2d = [](uint8_t* src, uint8_t* dst) {
    memcpy(dst, src, 224 * 224 * 3);
    return;
    // NCHW->NHWC
    // for (int i = 0; i < 3; i++) {
    //   for (int j = 0; j < 224 * 224; j++) {
    //     dst[j * 3 + i] = src[i * 224 * 224 + j];
    //   }
    // }
  };

  // auto show = [](cv::Mat img, int h, int w) {
  //   for (int i = 0; i < h; i++) {
  //     for (int j = 0; j < w; j++) {
  //       printf("%d %d %d\n", ((uint8_t*)img.data)[i * w * 3 + j * 3 + 0],
  //              ((uint8_t*)img.data)[i * w * 3 + j * 3 + 1],
  //              ((uint8_t*)img.data)[i * w * 3 + j * 3 + 2]);
  //     }
  //   }
  //   exit(0);
  // };

  float* mat_data = (float*)malloc(224 * 224 * 3 * sizeof(float));
  cv::Mat source_o, img_o, img_r;
  source_o = cv::imread(file_name);
  cv::cvtColor(source_o, img_o, cv::COLOR_BGR2RGB);
  cv::resize(img_o, img_r, {224, 224});
  // show(img_r, 256, 256);

  uint8_t* trans_data = (uint8_t*)malloc(224 * 224 * 3 * sizeof(uint8_t));
  // NCHW -> NHWC
  transpose2d((uint8_t*)img_r.data, trans_data);

  for (int i = 0; i < 224; i++) {
    for (int j = 0; j < 224; j++) {
      mat_data[i * 224 * 3 + j * 3 + 0] =
          ((trans_data[i * 224 * 3 + j * 3 + 0] / 255.0) - 0.485) / 0.229;  // R
      mat_data[i * 224 * 3 + j * 3 + 1] =
          ((trans_data[i * 224 * 3 + j * 3 + 1] / 255.0) - 0.456) / 0.224;  // G
      mat_data[i * 224 * 3 + j * 3 + 2] =
          ((trans_data[i * 224 * 3 + j * 3 + 2] / 255.0) - 0.406) / 0.225;  // B
    }
  }
  return mat_data;
}

void ShowResult(float* res) {
  int n_ele = 1000;
  std::vector<std::pair<float, int>> sort_pairs;
  for (int i = 0; i < n_ele; ++i) {
    sort_pairs.emplace_back(res[i], i);
  }
  std::sort(sort_pairs.begin(), sort_pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
              return a.first > b.first;
            });
  auto labels = load_imagenet_labels();
  const int topk = 5;
  std::cout << ">>> Result:" << std::endl;
  for (int i = 0; i < topk; ++i) {
    std::cout << "top " << (i + 1) << " " << sort_pairs[i].first << " -> Index=["
              << sort_pairs[i].second << "]"
              << ", Label=[" << labels[sort_pairs[i].second] << "]" << std::endl;
    ;
  }
}

int GetTime() {
  int timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  return timestamp;
}

int main() {
  PreLoadParams();
  const auto& files = GetFileName();
  int total_time = 0;
  for (auto it : files) {
    std::cout << "\nBegin to predict : " << it << std::endl;
    auto img = PreProcess(it);

    int start = GetTime();

    int h0, w0, c0;
    int h1, w1, c1;
    img = ComputeLayerConv2d(img, 224, 224, h1, w1, c1, "conv1");
    img = ComputeLayerBatchNorm(img, h1, w1, c1, "bn1");
    img = ComputeLayerRelu(img, h1 * w1 * c1);
    img = ComputeLayerMaxPool(img);
    // layer1
    img = ComputeBottleNeck(img, 56, 56, h1, w1, c1, "layer1_bottleneck0", true);
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer1_bottleneck1", false);
    img = ComputeBottleNeck(img, h0, w0, h1, w1, c1, "layer1_bottleneck2", false);
    // layer2
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer2_bottleneck0", true);
    img = ComputeBottleNeck(img, h0, w0, h1, w1, c1, "layer2_bottleneck1", false);
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer2_bottleneck2", false);
    img = ComputeBottleNeck(img, h0, w0, h1, w1, c1, "layer2_bottleneck3", false);
    // layer3
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer3_bottleneck0", true);
    img = ComputeBottleNeck(img, h0, w0, h1, w1, c1, "layer3_bottleneck1", false);
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer3_bottleneck2", false);
    img = ComputeBottleNeck(img, h0, w0, h1, w1, c1, "layer3_bottleneck3", false);
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer3_bottleneck4", false);
    img = ComputeBottleNeck(img, h0, w0, h1, w1, c1, "layer3_bottleneck5", false);
    // layer4
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer4_bottleneck0", true);
    img = ComputeBottleNeck(img, h0, w0, h1, w1, c1, "layer4_bottleneck1", false);
    img = ComputeBottleNeck(img, h1, w1, h0, w0, c0, "layer4_bottleneck2", false);
    // avg pool
    img = ComputeLayerAvgPool(img);
    // Linear
    img = ComputeLayerFC(img, "fc");

    int end = GetTime();
    int time = end - start;
    total_time += time;
    ShowResult(img);

    std::cout << "Time cost : " << time << " ms.\n" << std::endl;
    free(img);
  }
  float latency = (float)(total_time) / (float)(files.size());
  std::cout << "\033[0;32mAverage Latency : " << latency << "ms \033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput : " << (1000 / latency) << "fps \033[0m" << std::endl;
  for (auto it : __global_params) {
    free(it.second);
  }
  return 0;
}
