#pragma once

float* MyConv2dPreLoad(float* img,
                       float* weight,
                       int hi,
                       int wi,
                       int& ho,
                       int& wo,
                       int ci,
                       int co,
                       int kernel,
                       int stride,
                       int pad);

float* MyConv2d(float* img,
                float* weight,
                int hi,
                int wi,
                int& ho,
                int& wo,
                int ci,
                int co,
                int kernel,
                int stride,
                int pad,
                bool First,
                bool is_free_img = true);
