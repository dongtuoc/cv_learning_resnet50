#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* MyFCPreLoad(float* img, float* weight, float* bias);

float* MyFC(float* img, float* weight, float* bias);
