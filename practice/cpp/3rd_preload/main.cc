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
#include "../utils.h"
#include "resnet_preload.h"

// 定义一个全局参数map，用于存储网络层的参数
// 该参数替代了 2nd_avx2 中在加载参数时，动态分配内存的操作，性能更好
extern std::map<std::string, void*> __global_params;

// 主函数入口
int main() {
  // 预加载网络参数
  PreLoadParams();

  // 获取图片文件名和对应的标签
  const auto& files = GetFileName();
  int total_time = 0;  // 耗时计数器

  // 遍历每个文件进行预测
  for (auto it : files) {
    std::cout << "\nBegin to predict : " << it.first << std::endl;  // 打印正在处理的文件名
    auto img = PreProcess(it.first);                                // 图像预处理

    // 获取预测开始时间
    int start = GetTime();

    // 定义存储层输出维度的变量
    int h0, w0, c0;
    int h1, w1, c1;

    // 依次通过卷积层、批归一化层、ReLU激活层、最大池化层处理图像
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

    // 计算预测结束时间和耗时
    int end = GetTime();
    int time = end - start;
    total_time += time;

    // 展示预测结果
    int res_label = ShowResult(img);
    // 判断预测结果是否正确
    if (res_label == it.second) {
      std::cout << "\033[0;32mInference Result Succ \033[0m" << std::endl;
    } else {
      std::cout << "\033[0;31mInference Result Fail: Golden Label: " << it.second
                << ", Res Lable: " << res_label << "\033[0m" << std::endl;
    }
    std::cout << "\033[0;32mTime cost : " << time << " ms.\033[0m\n" << std::endl;

    // 释放分配的内存
    free(img);
  }

  // 计算平均延迟和吞吐量
  float latency = (float)(total_time) / (float)(files.size());
  std::cout << "\033[0;32mAverage Latency : " << latency << "ms \033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput : " << (1000 / latency) << "fps \033[0m" << std::endl;

  // 释放全局参数所占的内存
  for (auto it : __global_params) {
    free(it.second);
  }
  return 0;
}
