#include "fc.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 这段代码定义了一个自定义的全连接层（Fully Connected Layer，也称为密集层）函数 my_fc
// 它是神经网络中的一个常见算法。
// my_fc
// 函数接收一个输入数组（img），该数组是前一层的输出，以及与每个神经元相关联的权重数组（weight）和偏置数组（bias）。
// 函数通过加权求和的方式计算每个输出神经元的值，并在计算完成后释放输入、权重和偏置的内存。
float* my_fc(float* img, float* weight, float* bias) {
  // 分配输出的内存空间，在resnet50中的全连接层，输出有1000个神经元，这里就直接将参数写成了 1000
  float* out = (float*)malloc(1000 * sizeof(float));

  // 遍历输出层的每个神经元
  for (int i = 0; i < 1000; i++) {
    float sum_x = float(0);

    // 遍历输入层的每个神经元（resent50中内层累加维度为2048）
    for (int j = 0; j < 2048; j++) {
      auto l = img[j];                // 获取输入层的第j个神经元的值
      auto r = weight[i * 2048 + j];  // 获取权重矩阵中对应的值
      sum_x += l * r;  // 执行加权求和，这也是全连接层的核心计算逻辑，和卷积有点类似，都是乘累加操作
    }

    // 加上偏置值并存储到输出数组中
    out[i] = sum_x + bias[i];
  }

  free(img);
  free(weight);
  free(bias);

  // 返回全连接层的输出
  return out;
}
