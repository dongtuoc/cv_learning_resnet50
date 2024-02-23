#include <dirent.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "./label.h"

static inline std::map<std::string, int> GetFileName() {
  std::map<std::string, int> res;
  std::string dir_path("../../pics/ani_12/");
  DIR* dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    std::cerr << "Failed to open directory: " << dir_path << std::endl;
    exit(0);
  }
  dirent* entry;
  std::vector<std::string> filenames;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG) {
      filenames.push_back(dir_path + std::string(entry->d_name));
    }
  }
  closedir(dir);

  int label = 0;
  for (const auto& filename : filenames) {
    if (filename == "../../pics/ani_12/LaoHu.jpg") {
      label = 292;
    } else if (filename == "../../pics/ani_12/Ji.jpg") {
      label = 7;
    } else if (filename == "../../pics/ani_12/HouZi.jpg") {
      label = 373;
    } else if (filename == "../../pics/ani_12/TuZi.jpg") {
      label = 282;
    } else if (filename == "../../pics/ani_12/Yang.jpg") {
      label = 348;
    } else if (filename == "../../pics/ani_12/Niu.jpg") {
      label = 345;
    } else if (filename == "../../pics/ani_12/LaoShu.jpg") {
      label = 367;
    } else if (filename == "../../pics/ani_12/Ma.jpg") {
      label = 268;
    } else if (filename == "../../pics/ani_12/Zhu.jpg") {
      label = 341;
    } else if (filename == "../../pics/ani_12/Long.jpg") {
      label = 39;
    } else if (filename == "../../pics/ani_12/Gou.jpg") {
      label = 258;
    } else if (filename == "../../pics/ani_12/She.jpg") {
      label = 53;
    } else {
      std::cout << "No Golden Label for " << filename << std::endl;
    }
    res[filename] = label;
  }

  std::cout << "\033[0;32m\nLoaded Pics List: \033[0m" << std::endl;
  for (auto it : res) {
    std::cout << it.first << ", lable: " << it.second << std::endl;
  }
  std::cout << "\n" << std::endl;
  return res;
}

static inline int ShowResult(void* res0) {
  float* res = (float*)res0;
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
  }
  return sort_pairs[0].second;
}

static inline int GetTime() {
  int timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  return timestamp;
}

static inline float* PreProcess(const std::string& file_name, void *out = nullptr) {
  float *mat_data = nullptr;
  if (out) {
    mat_data =(float*)out;
  } else {
    mat_data = (float*)malloc(224 * 224 * 3 * sizeof(float));
  }
  cv::Mat source_o, img_o, img_r;
  source_o = cv::imread(file_name);
  cv::cvtColor(source_o, img_o, cv::COLOR_BGR2RGB);
  cv::resize(img_o, img_r, {224, 224});

  uint8_t* trans_data = (uint8_t*)malloc(224 * 224 * 3 * sizeof(uint8_t));
  memcpy(trans_data, img_r.data, 224 * 224 * 3);

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
