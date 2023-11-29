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
#include "./resnet_no_malloc.h"

static std::map<std::string, int> GetFileName() {
  std::map<std::string, int> res;
  std::string dir_path("../../pics/ani_12/");
  // filenames.push_back(dir_path + std::string("Niu.jpg"));
  // return filenames;
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

  std::cout << "Read Files:" << std::endl;
  for (auto it : res) {
    std::cout << it.first << ", lable: " << it.second << std::endl;
  }
  std::cout << "\n\n" << std::endl;
  return res;
}

static void PreProcess(const std::string& file_name, void* out) {
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

  auto show = [](cv::Mat img, int h, int w) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        printf("%d %d %d\n", ((uint8_t*)img.data)[i * w * 3 + j * 3 + 0],
               ((uint8_t*)img.data)[i * w * 3 + j * 3 + 1],
               ((uint8_t*)img.data)[i * w * 3 + j * 3 + 2]);
      }
    }
    exit(0);
  };

  float* mat_data = (float*)out;
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
  free(trans_data);
}

int ShowResult(void* res0) {
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
  const int topk = 1;
  std::cout << ">>> Result:" << std::endl;
  for (int i = 0; i < topk; ++i) {
    std::cout << "top " << (i + 1) << " " << sort_pairs[i].first << " -> Index=["
              << sort_pairs[i].second << "]"
              << ", Label=[" << labels[sort_pairs[i].second] << "]" << std::endl;
  }
  return sort_pairs[0].second;
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

  void* __global_mem_main0 = malloc(8 * 1024 * 1024);
  void* __global_mem_main1 = malloc(8 * 1024 * 1024);
  void* __global_mem_temp = malloc(8 * 1024 * 1024);

  for (auto it : files) {
    out_cnt = 0;
    std::cout << "\nBegin to predict : " << it.first << std::endl;
    PreProcess(it.first, __global_mem_main0);

    int h0, w0, c0;
    int h1, w1, c1;

    int start = GetTime();
    ComputeLayerConv2d<false>(__global_mem_main0, __global_mem_main1, 224, 224, h1, w1, c1,
                              "conv1");
    ComputeLayerBatchNorm<false>(__global_mem_main1, __global_mem_main0, h1, w1, c1, "bn1");
    ComputeLayerRelu<false>(__global_mem_main0, h1 * w1 * c1);
    ComputeLayerMaxPool<false>(__global_mem_main0, __global_mem_main1);
    // layer1
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, 56, 56, h1,
                             w1, c1, "layer1_bottleneck0", true);
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer1_bottleneck1", false);
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1,
                             w1, c1, "layer1_bottleneck2", false);
    // layer2
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer2_bottleneck0", true);
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1,
                             w1, c1, "layer2_bottleneck1", false);
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer2_bottleneck2", false);
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1,
                             w1, c1, "layer2_bottleneck3", false);
    // layer3
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer3_bottleneck0", true);
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1,
                             w1, c1, "layer3_bottleneck1", false);
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer3_bottleneck2", false);
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1,
                             w1, c1, "layer3_bottleneck3", false);
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer3_bottleneck4", false);
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1,
                             w1, c1, "layer3_bottleneck5", false);
    // layer4
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer4_bottleneck0", true);
    ComputeBottleNeck<false>(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1,
                             w1, c1, "layer4_bottleneck1", false);
    ComputeBottleNeck<false>(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0,
                             w0, c0, "layer4_bottleneck2", false);
    // avg pool
    ComputeLayerAvgPool<false>(__global_mem_main1, __global_mem_main0);
    // Linear
    ComputeLayerFC<false>(__global_mem_main0, __global_mem_main1, "fc");

    int end = GetTime();
    int time = end - start;
    total_time += time;

    int res_label = ShowResult(__global_mem_main1);
    if (res_label == it.second) {
      std::cout << "\033[0;32mInference Result Succ \033[0m" << std::endl;
    } else {
      std::cout << "\033[0;31mInference Result Fail: Golden Label: " << it.second
                << ", Res Lable: " << res_label << "\033[0m" << std::endl;
    }
    std::cout << "\033[0;32mTime cost : " << time << " ms.\033[0m\n" << std::endl;
  }
  float latency = (float)(total_time) / (float)(files.size());
  std::cout << "\033[0;32mAverage Latency : " << latency << "ms \033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput : " << (1000 / latency) << "fps \033[0m" << std::endl;
  for (int i = 0; i < MAX_MEM_NUM; i++) {
    if (__global_weight[i] != nullptr) free(__global_weight[i]);
  }
  free(__global_mem_main0);
  free(__global_mem_main1);
  free(__global_mem_temp);
  return 0;
}
