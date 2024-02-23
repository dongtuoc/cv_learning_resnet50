#pragma once

float* compute_relu_layer(float* img, int len);
float* compute_conv_layer(float* img,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& layer_name,
                          bool is_free_img = true);

float* compute_fc_layer(float* img, const std::string& layer_name);

float* compute_bn_layer(float* in_data, int h, int w, int c, const std::string& layer_name);

float* compute_maxpool_layer(float* in_data);

float* compute_avgpool_layer(float* in_data);

float* compute_bottleneck(float* in_data,
                          int hi,
                          int wi,
                          int& ho,
                          int& wo,
                          int& co,
                          const std::string& bottleneck_layer_name,
                          bool down_sample);
