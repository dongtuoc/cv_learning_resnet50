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
#include "./resnet.h"

std::vector<std::string> getFileName() {
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
    if (entry->d_type == DT_REG) {  // 如果是普通文件
      filenames.push_back(dir_path + std::string(entry->d_name));
    }
  }
  closedir(dir);
  for (const auto& filename : filenames) {
    std::cout << filename << std::endl;
  }
  return filenames;
}

float* preprocess(const std::string& file_name) {
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
  // printf("%d %d\n", source_o.rows, source_o.cols);
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

void show_res(float* res) {
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
  for (int i = 0; i < topk; ++i) {
    std::cout << "top " << (i + 1) << " " << sort_pairs[i].first << " -> Index=["
              << sort_pairs[i].second << "]"
              << ", Label=[" << labels[sort_pairs[i].second] << "]" << std::endl;
    ;
  }
}

auto getTime() {
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  return timestamp;
}

int main() {
  const auto& files = getFileName();
  for (auto it : files) {
    auto start = getTime();
    std::cout << "Predict : " << it << std::endl;
    auto img = preprocess(it);
    int h0, w0, c0;
    int h1, w1, c1;
    img = compute_conv_layer(img, 224, 224, h1, w1, c1, "conv1");
    img = compute_bn_layer(img, h1, w1, c1, "bn1");
    img = compute_relu_layer(img, h1 * w1 * c1);
    img = compute_maxpool_layer(img);
    // layer1
    img = compute_bottleneck(img, 56, 56, h1, w1, c1, "layer1_bottleneck0", true);
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer1_bottleneck1", false);
    img = compute_bottleneck(img, h0, w0, h1, w1, c1, "layer1_bottleneck2", false);
    // layer2
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer2_bottleneck0", true);
    img = compute_bottleneck(img, h0, w0, h1, w1, c1, "layer2_bottleneck1", false);
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer2_bottleneck2", false);
    img = compute_bottleneck(img, h0, w0, h1, w1, c1, "layer2_bottleneck3", false);
    // layer3
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer3_bottleneck0", true);
    img = compute_bottleneck(img, h0, w0, h1, w1, c1, "layer3_bottleneck1", false);
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer3_bottleneck2", false);
    img = compute_bottleneck(img, h0, w0, h1, w1, c1, "layer3_bottleneck3", false);
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer3_bottleneck4", false);
    img = compute_bottleneck(img, h0, w0, h1, w1, c1, "layer3_bottleneck5", false);
    // layer4
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer4_bottleneck0", true);
    img = compute_bottleneck(img, h0, w0, h1, w1, c1, "layer4_bottleneck1", false);
    img = compute_bottleneck(img, h1, w1, h0, w0, c0, "layer4_bottleneck2", false);
    // avg pool
    img = compute_avgpool_layer(img);
    // Linear
    img = compute_fc_layer(img, "fc");

    auto end = getTime();
    std::cout << "Time cost : " << end - start << " ms." << std::endl;
    show_res(img);
    free(img);
  }
  return 0;
}
