#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

float* LoadCon2dWeightPreLoad(const std::string& name, int len);

float* LoadCon2dWeight(const std::string& name, int len);

int* LoadCon2dParamPreLoad(const std::string& name, int len);

int* LoadCon2dParam(const std::string& name, int len);

float* ComputeLayerReluPreLoad(float* img, int len);

float* ComputeLayerRelu(float* img, int len);

float* ComputeLayerConv2dPreLoad(float* img,
                                 int hi,
                                 int wi,
                                 int& ho,
                                 int& wo,
                                 int& co,
                                 const std::string& layer_name,
                                 bool is_free_img = true);

float* ComputeLayerConv2d(float* img,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& layer_name,
                          bool is_free_img = true);

float* ComputeLayerFCPreLoad(float* img, const std::string& layer_name);

float* ComputeLayerFC(float* img, const std::string& layer_name);

float* ComputeLayerBatchNormPreLoad(
    float* in_data, int h, int w, int c, const std::string& layer_name);

float* ComputeLayerBatchNorm(float* in_data, int h, int w, int c, const std::string& layer_name);

float* ComputeLayerMaxPoolPreLoad(float* in_data);

float* ComputeLayerMaxPool(float* in_data);

float* ComputeLayerAvgPoolPreLoad(float* in_data);

float* ComputeLayerAvgPool(float* in_data);

float* ComputeBottleNeckPreLoad(float* in_data,
                                int hi,
                                int wi,
                                int& ho,
                                int& wo,
                                int& co,
                                const std::string& bottleneck_layer_name,
                                bool down_sample);

float* ComputeBottleNeck(float* in_data,
                         int hi,
                         int wi,
                         int& ho,
                         int& wo,
                         int& co,
                         const std::string& bottleneck_layer_name,
                         bool down_sample);

void PreLoadParams();
