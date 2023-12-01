#pragma once

#include <string>

#define MAX_MEM_NUM (1024)

float* LoadCon2dWeightPreLoad(const std::string& name, int len);

float* LoadCon2dWeight();

int* LoadCon2dParamPreLoad(const std::string& name, int len);

int* LoadCon2dParam();

void ComputeLayerReluPreLoad(void* img_in, int len);

void ComputeLayerRelu(void* img_in, int len);

void ComputeLayerConv2dPreLoad(void* img_in,
                               void* img_out,
                               int hi,
                               int wi,
                               int& ho,
                               int& wo,
                               int& co,
                               const std::string& layer_name);

void ComputeLayerConv2d(void* img_in, void* img_out, int hi, int wi, int& ho, int& wo, int& co);

void ComputeLayerFCPreLoad(void* img_in, void* img_out, const std::string& layer_name);

void ComputeLayerFC(void* img_in, void* img_out);

void ComputeLayerBatchNormPreLoad(
    void* in_data, void* out_data, int h, int w, int c, const std::string& layer_name);

void ComputeLayerBatchNorm(void* in_data, void* out_data, int h, int w, int c);

void ComputeLayerMaxPoolPreLoad(void* in_data, void* out_data);

void ComputeLayerMaxPool(void* in_data, void* out_data);

void ComputeLayerAvgPoolPreLoad(void* in_data, void* out_data);

void ComputeLayerAvgPool(void* in_data, void* out_data);

void AddPreLoad(float* l, float* r, float* out, int len);

void Add(float* l, float* r, float* out, int len);

void ComputeBottleNeckPreLoad(void* in_data,
                              void* out_data,
                              void* temp_data,
                              int hi,
                              int wi,
                              int& ho,
                              int& wo,
                              int& co,
                              const std::string& bottleneck_layer_name,
                              bool down_sample);

void ComputeBottleNeck(void* in_data,
                       void* out_data,
                       void* temp_data,
                       int hi,
                       int wi,
                       int& ho,
                       int& wo,
                       int& co,
                       bool down_sample);
void PreLoadParams();

void CodeGen(void*, void*, void*);
