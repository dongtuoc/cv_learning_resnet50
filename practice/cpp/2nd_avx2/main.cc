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
#include "resnet_avx2.h"

// 主函数入口
// 这段代码展示了如何在 C++ 中利用加载的权重和网络结构进行图像分类的计算过程。
// 它还包括了性能度量，如平均延迟和吞吐量的计算
int main() {
  // 获取需要预测的图片和标签
  const auto& files = GetFileName();
  int total_time = 0;  // 总耗时清零

  // 遍历每个文件
  for (auto it : files) {
    std::cout << "Predict : " << it.first << std::endl;  // 打印正在处理的文件名

    // 图像预处理
    auto img = PreProcess(it.first);
    int h0, w0, c0;  // 用于存储层输出的尺寸
    int h1, w1, c1;

    // 通过网络层处理图像
    auto start = GetTime();  // 开始时间
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
    // 结束时间和耗时计算
    auto end = GetTime();
    int time = end - start;
    total_time += time;

    // 展示结果
    int res_label = ShowResult(img);
    // 检查预测结果是否与标签一致
    if (res_label == it.second) {
      std::cout << "\033[0;32mInference Result Succ \033[0m" << std::endl;
    } else {
      std::cout << "\033[0;31mInference Result Fail: Golden Label: " << it.second
                << ", Res Lable: " << res_label << "\033[0m" << std::endl;
    }
    std::cout << "\033[0;32mTime cost : " << time << " ms.\033[0m\n" << std::endl;

    // 释放内存
    free(img);
  }

  // 计算平均延迟和吞吐量
  float latency = (float)(total_time) / (float)(files.size());
  std::cout << "\033[0;32mAverage Latency : " << latency << "ms \033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput : " << (1000 / latency) << "fps \033[0m" << std::endl;
  return 0;
}