#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* MyBatchNormPreLoad(
    float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c);

float* MyBatchNorm(
    float* img, float* mean, float* var, float* gamma, float* bias, int h, int w, int c);
