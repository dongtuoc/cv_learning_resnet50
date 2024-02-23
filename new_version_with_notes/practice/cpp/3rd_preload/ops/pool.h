#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* MyMaxPoolPreLoad(float* img);

float* MyMaxPool(float* img);

float* MyAvgPoolPreLoad(float* img);

float* MyAvgPool(float* img);
