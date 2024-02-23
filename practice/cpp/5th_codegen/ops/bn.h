#pragma once

void MyBatchNormPreLoad(void* img_in,
                        void* img_out,
                        float* mean,
                        float* var,
                        float* gamma,
                        float* bias,
                        int h,
                        int w,
                        int c);

void MyBatchNorm(void* img_in,
                 void* img_out,
                 float* mean,
                 float* var,
                 float* gamma,
                 float* bias,
                 int h,
                 int w,
                 int c);
