#include <dirent.h>
#include <dlfcn.h>

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
#include "./resnet_codegen.h"
#include "ops/common.h"

extern int put_cnt;
extern int out_cnt;
extern void* __global_weight[MAX_MEM_NUM];

int main(int argc, char *argv[]) {
  int step_flag = 0;
  if (argc >= 2) {
    int num = std::atoi(argv[1]);
    if (num == 0) step_flag = 1;
    if (num == 1) step_flag = 2;
  }

  PreLoadParams();
  const auto& files = GetFileName();
  int total_time = 0;

  void* __global_mem_main0 = malloc(8 * 1024 * 1024);
  void* __global_mem_main1 = malloc(8 * 1024 * 1024);
  void* __global_mem_temp = malloc(8 * 1024 * 1024);

  // Once the input/output address of all layers are fixed, and the
  // params of all layers are determined, we can generate a `const` code
  // which have a better performence due to the following reasons.
  // 1. Uncessary calulating for all intermediate data and params is removed.
  // 2. All weight/bias/OtherParams are fixed to a determined address.
  //
  // CodeGen function will generate a `libcodegen.so` under ./codegen/ directory.
  CodeGen(__global_mem_main0, __global_mem_main1, __global_mem_temp);
  if (step_flag == 1) return 0;

  CompileModule();
  if (step_flag == 2) return 0;

  auto Resnet = (Module)LoadModule();

  for (auto it : files) {
    out_cnt = 0;
    std::cout << "\nBegin to predict : " << it.first << std::endl;
    PreProcess(it.first, __global_mem_main0);

    int start = GetTime();
    Resnet(__global_mem_main0, __global_mem_main1, __global_mem_temp);
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
