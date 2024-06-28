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
#include "./resnet_no_malloc.h"

extern int put_cnt;
extern int out_cnt;
extern void* __global_weight[MAX_MEM_NUM];

int main() {
  // 预加载神经网络的所有参数到 __global_weight 中
  PreLoadParams();
  // 获取文件名列表
  const auto& files = GetFileName();
  int total_time = 0;

  // 分配全局内存空间
  void* __global_mem_main0 = malloc(8 * 1024 * 1024);
  void* __global_mem_main1 = malloc(8 * 1024 * 1024);
  void* __global_mem_temp = malloc(8 * 1024 * 1024);

  // 遍历每个文件进行推理预测
  for (auto it : files) {
    out_cnt = 0;  // 重置输出计数器，用来对从 __global_weight 里取权值进行计数操作
    std::cout << "\nBegin to predict : " << it.first << std::endl;
    // 对原始输入图像进行预处理
    PreProcess(it.first, __global_mem_main0);

    int h0, w0, c0;
    int h1, w1, c1;

    int start = GetTime();  // 记录推理开始时间
    // 执行各层的计算
    ComputeLayerConv2d(__global_mem_main0, __global_mem_main1, 224, 224, h1, w1, c1);
    ComputeLayerBatchNorm(__global_mem_main1, __global_mem_main0, h1, w1, c1);
    ComputeLayerRelu(__global_mem_main0, h1 * w1 * c1);
    ComputeLayerMaxPool(__global_mem_main0, __global_mem_main1);
    // 执行残差块的计算
    // layer1
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, 56, 56, h1, w1, c1,
                      true);
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      false);
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1, w1, c1,
                      false);
    // layer2
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      true);
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1, w1, c1,
                      false);
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      false);
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1, w1, c1,
                      false);
    // layer3
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      true);
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1, w1, c1,
                      false);
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      false);
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1, w1, c1,
                      false);
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      false);
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1, w1, c1,
                      false);
    // layer4
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      true);
    ComputeBottleNeck(__global_mem_main1, __global_mem_main0, __global_mem_temp, h0, w0, h1, w1, c1,
                      false);
    ComputeBottleNeck(__global_mem_main0, __global_mem_main1, __global_mem_temp, h1, w1, h0, w0, c0,
                      false);
    // avg pool
    ComputeLayerAvgPool(__global_mem_main1, __global_mem_main0);
    // Linear
    ComputeLayerFC(__global_mem_main0, __global_mem_main1);

    int end = GetTime();     // 记录推理结束时间
    int time = end - start;  // 计算耗时
    total_time += time;      // 计算所有图片的总耗时，用于计算平均值

    int res_label = ShowResult(__global_mem_main1);  // 显示结果
    if (res_label == it.second) {
      std::cout << "\033[0;32mInference Result Succ \033[0m" << std::endl;
    } else {
      std::cout << "\033[0;31mInference Result Fail: Golden Label: " << it.second
                << ", Res Lable: " << res_label << "\033[0m" << std::endl;
    }
    std::cout << "\033[0;32mTime cost : " << time << " ms.\033[0m\n" << std::endl;
  }
  // 计算平均延迟和吞吐量
  float latency = (float)(total_time) / (float)(files.size());
  std::cout << "\033[0;32mAverage Latency : " << latency << "ms \033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput : " << (1000 / latency) << "fps \033[0m" << std::endl;
  // 释放内存资源
  for (int i = 0; i < MAX_MEM_NUM; i++) {
    if (__global_weight[i] != nullptr) free(__global_weight[i]);
  }
  free(__global_mem_main0);
  free(__global_mem_main1);
  free(__global_mem_temp);
  return 0;
}