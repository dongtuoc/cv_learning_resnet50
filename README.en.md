# cv_learning_resnet50
---
## Project Introduction

### What is this project doing?
This project aims to provide an introductory learning experience in computer vision for AI.

Initially, it involves practical exercises on classical computer vision algorithms to understand the fundamentals of computer vision. Subsequently, using the ResNet50 neural network as an example, the project systematically explains the basic algorithm principles and related background knowledge of an AI model.

Finally, through hands-on coding in this repository, the project involves writing a ResNet50 neural network from scratch to achieve image recognition for any given image, as well as performance optimization of the neural network model.

- The traditional computer vision part includes practical exercises such as grayscale images, RGB, mean/Gaussian filtering, edge detection using the Canny operator, and image segmentation using the Otsu algorithm.

- The AI part includes a handwritten digit recognition project (MNIST) as an introduction to understand the training and inference process of an AI model.

- The AI principles section elaborates on and analyzes the algorithms and background knowledge used in ResNet50.

- In the hands-on section, ResNet50 model is written from scratch in both Python and C++.

  - All core algorithms and network structures of ResNet50 (including Conv2d, AvgPool, MaxPool, FC, Relu, residual structure) are handwritten without relying on any third-party libraries.
  - Since the algorithms and model structures are handwritten, there is considerable flexibility for performance optimization. Performance optimization is the final part of the project, involving iterations of multiple versions to gradually improve the model's performance.
- In the practical section, besides ensuring the network's accuracy (i.e., correctly predicting the image's Top1/Top5 after preprocessing any given image), performance optimization is also emphasized, as described later.

- The hands-on coding part of C++ and the performance optimization sections rely on Intel CPUs. Why use a CPU? Because it's a chip that everyone can access (almost all computers have Intel CPUs), whereas many students may not have access to GPUs, and the programming difficulty of GPUs is also higher.

---
### Why write all core algorithms from scratch

Currently, there are many tutorials online that teach you how to build neural networks, mostly based on torch's nn module or other modules, where you can simply use nn.conv2d to perform convolution calculations.

For students who want to delve deeper into algorithms or learn the fundamentals, or for some beginners, even if they follow the tutorials to build neural networks or perform inference on images, they may still feel confused and unable to grasp the underlying principles. This was particularly evident to me when I first started learning many years ago.

In fact, nn.conv2d encapsulates the implementation of conv2d algorithm, making it difficult for us to see its implementation details. It's hard to learn the implementation details inside and even harder to perform performance optimization on top of it (although this interface has been optimized).

**So, this project came about.**

Initially, I just wanted to handwrite a simple ResNet50 model on my own.

Later, some friends contacted me hoping to learn along with me. So I began to systematically write articles, and the more I wrote, the more I was motivated to maintain and update this project. Up to now, I am still continuously updating the code, adding comments to the code, and writing related articles.

Therefore, you can see that while the code part of this project is available for everyone to download and learn from, there are also more than 100 articles accompanying the repository. If you're interested, you can [check them out here](https://mp.weixin.qq.com/s/UWdQmlCrnzUvw8kOIb3PXw).

The code in this project was started in April 2023, made into a booklet in November 2023, and has been debugged many times. All the code was handwritten by me.

Currently, all the code in the project runs smoothly, and the accuracy is also OK. After five iterations, the performance has also reached a good level.

### What You Can Learn

Through this project, you can gain insights into classical algorithms of traditional computer vision, understand the connection and differences between traditional computer vision and deep learning-based computer vision algorithms, delve into all the algorithm prototypes used in ResNet50, understand the background principles of these algorithms, grasp the concepts of ResNet50, comprehend its network structure, and learn about common neural network optimization methods.

You can refer to the code in the project to actually run a ResNet50 neural network and perform inference on one or multiple images.

If you go through the project code and accompanying articles thoroughly, and practice with them, I believe that getting started with AI vision is not difficult. Moreover, even if you are a beginner, after fully practicing with ResNet50, you can consider yourself proficient.

---
## Related Articles

This project is accompanied by over 100 articles covering background knowledge, theoretical analysis, and practical coding, which have been meticulously written.

There are two ways to access these articles:

1. Subscribe [here](https://mp.weixin.qq.com/s/UWdQmlCrnzUvw8kOIb3PXw).

2. Subscribe [here](https://mp.weixin.qq.com/s/9De_ys5ibl6JxEzFN_qZmQ).

---
## Repository Structure
- **0_gray**: Code related to grayscale images.
- **1_RGB**: Code related to converting between grayscale and RGB.
- **2_mean_blur**: Code related to mean blur filtering.
- **3_gaussian_blur**: Code related to Gaussian blur filtering.
- **4_canny**: Code related to the Canny algorithm, used for **edge detection** in images.
- **5_otsu**: Code related to the Otsu algorithm, used for image **segmentation**.
- **6_mnist**: A classic handwritten digit recognition AI model (neural network), capable of training and inference on a laptop (CPU).
- **practice**: Main directory for handwriting the ResNet50 model algorithm, building the model, and related components from scratch. It includes:
  - **model**: Files related to open-source models, including downloading model parameters and parsing parameters.
  - **pics**: Directory for storing images when using the model for image recognition.
  - **python**: ResNet50 project written in Python.
  - **cpp**: ResNet50 project written in C++.

The **python** and **cpp** directories are independent of each other.

Within the **cpp** directory, there are 6 directories labeled from **1st** to **6th**, representing iterations of performance optimization. Each directory can be run independently to compare the performance improvement achieved through code optimization during each iteration.

---
## How I Implemented ResNet50 from Scratch

### Implementation Approach

#### Model Acquisition

Using torchvision, the weights of each layer of ResNet50 from a pre-trained model are saved to the repository. These saved weight files will be loaded later and used in convolution, fully connected, and batch normalization layer calculations.

It's worth mentioning that in real-world model deployment for industrial projects, the neural network weights are also loaded as independent data into GPU/CPU memory for computation.

However, the loading of model weights is often a bottleneck for performance in practical models. Why? I analyze several reasons:
- Limited by chip memory. It may not be possible to load all the weights of a neural network at once, resulting in multiple loading operations, which in turn bring unnecessary IO operations. This issue becomes more severe as memory becomes smaller.
- Limited by chip bandwidth. With the increasing size of model parameters today, GB-level bandwidth is becoming more and more strained. Moreover, in many cases, IO and computation cannot fully flow on the chip, especially in the case of heap computing, where IO becomes more prominent.

Under the **model** directory, running the following script will save the parameters to the **model/resnet50_weight** directory:
```python
$ python3 resnet50_parser.py
```

#### Code Implementation

After saving the weights, core functions such as Conv2d, BatchNorm, Relu, AvgPool, MaxPool, FullyConnect (MatMul), etc., are implemented separately using Python/C++.

Following the [ResNet50 network structure](https://mp.weixin.qq.com/s/nIeu58iSy5-cTafkvFxdig), these algorithms are pieced together.
- For the model files, refer to [model/resnet50.onnx.png](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/model/resnet50.onnx.png) and [model/resnet50_structure.txt](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/model/resnet50_structure.txt).
- For the manually constructed ResNet50 network structure, refer to [the Python version I manually built](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/python/infer.py#L354).

#### Inference

After implementing the code, implying that the basic algorithm and parameters required for model operation are in place, the next step is to read a local image and perform inference.
- Read an image of a cat, refer to [Get Image](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/python/infer.py#L296).

After reading the image, inference begins, and correctly inferring it as a cat completes the primary goal of this project (accuracy verification).

#### Optimization

Once the basic functionalities are implemented, efforts are made to optimize performance.

Performance optimization is a major focus in neural networks, and a separate section will explain this further.

---
## Performance Optimization

### Python Version

This section covers performance optimization for the Python version. Let's first understand how to use the Python code in this repository.

#### How to Use the Python Version

1. The core algorithms and handwritten networks of ResNet50 are written in basic Python syntax, with some basic operations using the numpy library.
2. Image importing uses the Pillow library. Image importing logic does not fall under the scope of **handwriting ResNet50 core algorithms**. Therefore, I didn't write similar logic from scratch and directly used the Pillow library.
3. To install dependencies, mainly the above two libraries, run the following command in the python directory:

Without using Tsinghua source:
```shell
$ pip3 install -r requirements.txt
```

Using Tsinghua source:
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. Inference
- In the python directory, run the following command to complete inference. You can modify the logic in my_infer.py to replace the image with your own and see if it can correctly recognize it.
```shell
$ python3 my_infer.py
```

Since the Python version also barely calls third-party libraries and writes convolution loops using Python syntax, its performance is extremely poor. In practice, inferring one image using the Python version is very slow, mainly due to excessive looping (but it's intended to demonstrate the internal implementation of the algorithm).

#### A Bit of Optimization for the Python Version

Use np.dot (inner product) instead of convolution's multiply-accumulate loop.
- Optimized version of the Python implementation: [Optimized Version](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/python/ops/conv2d.py#L73)

Since Python does not call third-party libraries, many optimization points cannot be addressed (such as controlling instruction sets and memory). Below, I will focus on optimizing the C++ version.

---
### C++ Version

This section covers performance optimization for the C++ version. Let's first understand how to use the C++ code in this repository.

#### How to Use the C++ Version

The C++ code in this repository has been integrated with several optimization commits, each time building on the previous optimization. You can easily see the optimization records through the filenames under the cpp directory.

- [cpp/1st_origin](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/1st_origin): Contains the code for the first version of C++.
- [cpp/2nd_avx2](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/2nd_avx2): Contains the code for the second version of C++, which enables **AVX2 instruction set optimization** and uses the **-Ofast compilation option**.
- [cpp/3rd_preload](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/3rd_preload): Contains the code for the third version of C++, which utilizes a memory pool-like approach to add **preloading logic for weights**, while still maintaining dynamic malloc for input and output of each layer's results.
- [cpp/4th_no_malloc](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/4th_no_malloc): Contains the code for the fourth optimized version of C++, which removes all dynamic memory allocation operations, significantly improving performance.
- [cpp/5th_codegen](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/5th_codegen): Contains the code for the fifth optimized version of C++, which utilizes CodeGen and jit compilation techniques to generate core calculation logic.
- [cpp/6th_mul_thread](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/6th_mul_thread): Contains the code for the sixth optimized version of C++, which utilizes multithreading to optimize convolution operations, greatly improving performance.

#### Compilation

The files under each version's directory are independent, with no dependencies between them. If you want to see the code changes between two versions, you can use a source code comparison tool to view them.

The compilation process for each version's directory is the same. If you only have a Windows environment without a Linux environment, you can quickly install a Linux system [here](https://mp.weixin.qq.com/s/iCdgm_vBnOFGa5b6yISKVA) without using a virtual machine in just 10 minutes. If you have purchased a paid article, you will have more detailed installation instructions.

If you have a Linux environment and are familiar with Linux operations, please proceed below:

- The C++ version compilation depends on the OpenCV library for image importing, which performs a similar function to the Pillow library in the Python version. In a Linux environment, run the following command to install the OpenCV library:
```shell
$ sudo apt-get install libopencv-dev python3-opencv libopencv-contrib-dev
```
- Under the cpp directory, run [compile.sh](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/1st_origin/compile.sh) to compile the files.
```shell
$ bash ./compile.sh
```
After compilation, a executable file named **resnet** will be generated in the current directory. Simply execute this file to perform inference on the images stored in the repository and display the results.
```shell
$ ./resnet
```

### Initial Version One

Directory: [cpp/1st_origin](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/1st_origin).

The first version did not consider performance issues, merely completing the functionality according to the idea. As a result, the performance was extremely poor. Here are the performance metrics for this version:

| Average Latency | Average Throughput |
| ---- | ---- |
| 16923 ms | 0.059 fps |

Performance data may vary depending on the computer's performance. You can try running it yourself to see the latency printed out.

### Optimized Version Two

Directory: [cpp/2nd_avx2](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/2nd_avx2).

The second version, based on the first version, parallelizes the **multiply-accumulate** loop operations in the convolution algorithm using vector instruction sets for accelerated parallelization. It utilizes the AVX2 vector instruction set. You can check if your CPU supports the AVX2 instruction set with the following command:

```shell
$ cat /proc/cpuinfo
```

If AVX2 is listed in the displayed information, your CPU supports this instruction set.

Here are the performance metrics for this version:

| Average Latency | Average Throughput |
| ---- | ---- |
| 4973 ms | 0.201 fps |

### Optimized Version Three

Directory: [cpp/3rd_preload](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/3rd_preload).

The third version, based on the second version, eliminates the dynamic malloc process for weight parameters during inference. Instead, before inference, it manages a class memory pool structure using std::map to preload all weight parameters. This optimization has practical significance in actual model deployment.

Preloading model parameters can maximize system IO pressure reduction and reduce latency.

Here are the performance metrics for this version:

| Average Latency | Average Throughput |
| ---- | ---- |
| 862 ms | 1.159 fps |

### Optimized Version Four

Directory: [cpp/4th_no_malloc](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/4th_no_malloc).

The fourth version, based on the third version, eliminates all dynamic memory allocation operations and string-related operations during inference.

Here are the performance metrics for this version:

| Average Latency | Average Throughput |
| ---- | ---- |
| 742 ms | 1.347 fps |

### Optimized Version Five

Directory: [cpp/5th_codegen](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/5th_codegen).

The fifth version, based on the fourth version, utilizes CodeGen technology to generate core calculation logic and completes the compilation process using jit compilation.

Here are the performance metrics for this version:

| Average Latency | Average Throughput |
| ---- | ---- |
| 781 ms | 1.281 fps |

### Optimized Version Six

Directory: [cpp/6th_mul_thread](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/practice/cpp/6th_mul_thread).

The sixth version, based on the fifth version, optimizes convolution calculations using multithreading, independently splitting threads for the co dimension, using all CPU threads.

Here are the performance metrics for this version:

| Average Latency | Average Throughput |
| ---- | ---- |
| 297 ms | 3.363 fps |

After six rounds of optimization, the inference latency has been improved from 16923 ms to 297 ms, nearly a 60-fold improvement in performance. Inferring one image no longer feels laggy, which is considered a good result.

---
## Overall Repository Dependencies

1. Dependencies for Saving Model Weights
- Navigate to the model directory and install dependencies for parsing models.
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. Dependencies for Python Inference

- Navigate to the python directory and install dependencies for inferring ResNet50, mainly numpy and Pillow libraries for importing images.
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---
## Contact Me for Other Inquiries

- All code and related articles in this project are original works. Please do not reprint them on any platform without permission, especially for commercial purposes. I have authorized relevant rights protection personnel to supervise the original articles and code.
- If you have any other inquiries, feel free to contact me (WeChat: ddcsggcs, or email: dongtuoc@yeah.net).

---
## May everyone be able to quickly get started with AI vision.




