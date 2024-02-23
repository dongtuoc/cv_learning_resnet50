#include <dirent.h>  // 引入目录操作的标准库，用于文件和目录的处理

#include <chrono>    // 引入时间库，用于进行时间的测量和转换
#include <cmath>     // 引入数学库，提供基本的数学运算功能
#include <cstdint>   // 引入标准整数类型的库，定义了固定大小的整数类型
#include <cstring>   // 引入C风格字符串处理库，提供字符串操作的函数
#include <iostream>  // 引入标准输入输出流库，用于输入输出操作
#include <opencv2/opencv.hpp>  // 引入OpenCV库，用于图像处理和计算机视觉
#include <string>              // 引入字符串库，提供C++风格的字符串操作功能
#include <vector>              // 引入向量库，提供动态数组的功能

#include "./label.h"  // 引入本地目录下的自定义头文件label.h，用于标签处理或定义

// 这个函数的主要功能是读取指定目录下的所有文件
// 并根据文件名给每个文件分配一个预定义的标签（例如，不同动物的图片分配不同的标签）。
// 函数最后返回一个包含文件名和对应标签的 map。
static inline std::map<std::string, int> GetFileName() {
  std::map<std::string, int> res;  // 创建一个字符串到整数的map，用于存储文件名和标签

  std::string dir_path("../../pics/ani_12/");  // 指定目录路径
  DIR* dir = opendir(dir_path.c_str());        // 打开目录
  if (dir == nullptr) {                        // 检查目录是否成功打开
    std::cerr << "Failed to open directory: " << dir_path << std::endl;  // 打印错误信息
    exit(0);  // 未能打开目录时退出程序
  }

  dirent* entry;                       // 定义一个目录项指针
  std::vector<std::string> filenames;  // 用于存储文件名的 vector

  // 读取目录中的每个文件
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG) {                                 // 检查是否为普通文件
      filenames.push_back(dir_path + std::string(entry->d_name));  // 将文件名添加到vector中
    }
  }
  closedir(dir);  // 关闭目录

  int label = 0;  // 初始化标签为0
  // 遍历文件名vector，并根据文件名分配相应的标签
  for (const auto& filename : filenames) {
    // 以下是一系列的条件判断，为特定的文件名分配标签
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
    res[filename] = label;  // 将文件名和对应的标签存储在映射中
  }

  // 打印加载的文件名和标签列表
  std::cout << "\033[0;32m\nLoaded Pics List: \033[0m" << std::endl;
  for (auto it : res) {
    std::cout << it.first << ", lable: " << it.second << std::endl;
  }
  std::cout << "\n" << std::endl;

  return res;  // 返回文件名和标签的 map
}

// 静态内联函数，用于显示模型预测结果
static inline int ShowResult(void* res0) {
  float* res = (float*)res0;  // 将void指针转换为float指针
  int n_ele = 1000;  // 设置结果元素的数量, resnet50 模型对应的训练数据集分类总共1000个
  std::vector<std::pair<float, int>> sort_pairs;  // 存储分数和索引的vector

  // 遍历每个元素，将分数和索引放入sort_pairs中
  for (int i = 0; i < n_ele; ++i) {
    sort_pairs.emplace_back(res[i], i);
  }

  // 对结果进行排序，按照分数从高到低
  std::sort(sort_pairs.begin(), sort_pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
              return a.first > b.first;
            });

  auto labels = load_imagenet_labels();  // 加载ImageNet标签, 函数定义在 label.h 中
  const int topk = 5;  // 设置要显示的前N个预测结果，这里设置显示 top5

  // 打印结果
  std::cout << ">>> Result:" << std::endl;
  for (int i = 0; i < topk; ++i) {
    std::cout << "top " << (i + 1) << " " << sort_pairs[i].first << " -> Index=["
              << sort_pairs[i].second << "]"
              << ", Label=[" << labels[sort_pairs[i].second] << "]" << std::endl;
  }
  return sort_pairs[0].second;  // 返回最高分数的索引
}

// 静态内联函数，用于获取当前时间的毫秒数
static inline int GetTime() {
  int timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();  // 获取当前时间戳的毫秒数
  return timestamp;              // 返回时间戳
}

// 预处理函数
// 输入为图像的路径，函数内部会加载图像，进行预处理
static inline float* PreProcess(const std::string& file_name, void* out = nullptr) {
  float* mat_data = nullptr;
  if (out) {
    mat_data = (float*)out;
  } else {
    mat_data = (float*)malloc(224 * 224 * 3 * sizeof(float));
  }
  cv::Mat source_o, img_o, img_r;
  source_o = cv::imread(file_name);  // 加载图像
  // opencv 默认是 BGR 的方式存储，转换为RGB通道来进行预处理
  cv::cvtColor(source_o, img_o, cv::COLOR_BGR2RGB);
  cv::resize(img_o, img_r, {224, 224});  // 将图像长宽 resize 到 224x224

  uint8_t* trans_data = (uint8_t*)malloc(224 * 224 * 3 * sizeof(uint8_t));
  memcpy(trans_data, img_r.data, 224 * 224 * 3);

  // 对图像的每个通道进行归一化，这里用到的归一化数值是预训练模型训练时使用的数值
  // 在推理时用和训练一样的数值，可以确保模型具有较好的精度
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
  return mat_data;  // 返回经过预处理之后的图像数据
}
