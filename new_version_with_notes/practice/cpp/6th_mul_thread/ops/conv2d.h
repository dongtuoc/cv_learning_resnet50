#pragma once

void MyConv2dPreLoad(void* img_in,
                     void* img_out,
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

void MyConv2d(void* img_in,
              void* img_out,
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
              bool First);
