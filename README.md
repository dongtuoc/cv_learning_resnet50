# cv_learning_resnet50

If you cannot understand Chinese, please refer to [this section](https://github.com/dongtuoc/cv_learning_resnet50/blob/main/README.en.md).


---
## 项目介绍

### 本项目在做什么

本项目旨在完成对 AI 的计算机视觉的入门学习。

首先通过对一些经典的传统计算机视觉算法进行实操，理解计算机视觉的含义；随后以 resnet50 神经网络为例子，系统的讲解一个 AI 模型的基础算法原理和相关背景知识。

最后通过本仓库中的代码实战，从零手写 resnet50 神经网络，完成任意一张图片的识别，以及神经网络模型的**性能优化**。

- 传统计算机视觉部分，会有灰度图、RGB、均值/高斯滤波器、利用 Canny 算子对图像进行边缘检测、利用大津算法对图像进行分割等小的实操项目。

- AI 部分会有一个手写数字识别项目(Mnist) 作为引子，来理解一个 AI 模型的训练和推理过程。

- AI 原理部分，会详细阐述和解析 resnet50 中用到的算法和背景知识。

- 实战部分用 python/C++ 两种语言完成 resnet50 模型的从零手写。
  - 其中 resnet50 的所有核心算法和网络结构(包括Conv2d、AvgPool、MaxPool、fc、Relu、残差结构) 全部手写，不借用任何第三方库。
  - 由于是自己手写的算法和模型结构，因此会有很大的自由度进行性能优化，性能优化是本项目最后进行的部分，会迭代多个版本，一步步将模型的性能调到比较不错的效果。

- 实战部分在完成手写算法的基础上，除了要保证网络精度可用(也就是任意给一张图片，经过预处理之后，Top1/Top5 可以正确的预测出图片) 之外，还会关注性能优化部分，这一点后面会有介绍。

- 代码实战的 C++ 部分以及性能优化部分，依赖于 Intel CPU 进行。为什么用 CPU？因为这是大家都能接触到的一个芯片(有电脑基本都会有 Intel CPU)，而 GPU 很多同学接触不到，并且 GPU 的编程难度也会大一些。

---
### 为什么要全部手写核心算法

目前网上有很多教程，在教你手搭神经网络的时候，基本都是基于 torch 的 nn 模块或其他模块，用 nn.conv2d 就完成了卷积计算。

对于想深究算法和学习算法的同学，或者一些初学者而言，即使按照教程将神经网络搭建出来了，或这将图片推理出来了，依旧是云里雾里，不知其原理，始终浮于表面，心里学的也不踏实，这一点我在多年前初学的时候感受尤为明显。

事实上，nn.conv2d 是将 conv2d 的算法实现给封装起来了，我们看不到它的实现机制，很难学到里面的实现细节，跟别提如何在此基础上进行性能优化了(虽然该接口已经被优化过)。

**于是便有了这个项目。**

最初仅仅是想自己动手完成一个简单的 resnet50 的模型的手写。

随后有一些小伙伴联系我希望跟着学习，于是开始系统的写文章，结果越写越多，索性做了一个小册，通过小册的写作，激励我不断的维护和更新这个项目，截止到现在，还在不断的更新代码，为代码写注释，写相关的文章。

所以，你可以看到，本项目的代码部分是大家都可以下载学习的，但是仓库配套的 100 多篇文章是付费的，如果你感兴趣，可以[来这里看看](https://mp.weixin.qq.com/s/UWdQmlCrnzUvw8kOIb3PXw)。

该项目中的代码从2023年4月开始编写，2023年11月做成小册，陆续调试了很多次，所有代码都是我手动编写而成。

目前项目中所有代码已经完全跑通，精度也很OK，性能经过 5 个版本的迭代，也基本达到了不错的效果。

### 你可以学到什么

通过本项目，你可以一窥传统计算机视觉的经典算法，理解传统计算机视觉和基于深度学习的计算机视觉算法的联系和区别，深入理解 resnet50 中用到的所有算法原型、算法背景原理、resent50 的思想、resnet50 的网络结构、以及常见的神经网络优化方法。

你可以参考项目中的代码，真正运行一个 resnet50 神经网络，完成一张或多张图片的推理。

在项目的 new_version_with_notes 目录下是带有**注释**的版本，代码中关键的地方我会给出文字详解。

如果你把项目代码和配套的文章都阅读一遍，完全实操一遍，我觉得入门 AI 视觉并不是难事，同时关于 resnet50 这个经典模型，即使你是一个小白，完全实操一遍之后也可以出师了。

---
## 项目所涉及文章

该项目搭配有 100+ 篇背景知识、原理解析和代码实操相关的介绍文章, 花费了巨大的精力写成。

有两种办法可以阅读到这些文章：

1. 在[这里](https://mp.weixin.qq.com/s/UWdQmlCrnzUvw8kOIb3PXw)订阅。

2. 在[这里](https://mp.weixin.qq.com/s/9De_ys5ibl6JxEzFN_qZmQ)订阅。


---
## 仓库结构
- 0_gray 为灰度图相关代码
- 1_RGB 为灰度图与 RGB 转换相关代码
- 2_mean_blur 为均值滤波相关代码
- 3_gussian_blur 为高斯滤波相关代码
- 4_canny 为 canny 算法相关，用来完成图片的**边缘检测**
- 5_dajin 为大津算法相关，用来完成图片的**分割**
- 6_minst 为一个经典的手写数字识别 AI 模型(神经网络)，可以在笔记本(CPU)上进行模型的训练和推理
- practice 为以 resnet50 为基础的模型算法手写、模型搭建和相关的主目录，也是本项目从零手写 resnet50 的主要目录，这里面又包含了：
  - model 目录：与开源模型相关的文件，包括模型参数的下载，参数的解析等。
  - pics 目录: 使用模型识别一张图片时，存放图片的目录
  - python 目录：利用 python 语言手写的 resnet50 项目
  - cpp 目录：利用 c++ 语言手写的 resnet50 项目。

其中，python 目录和 cpp 目录互相独立。

在 cpp 目录中，分别存在 1st 到 6th 6个目录，为性能优化的迭代目录，6 个目录相互独立，可以独立运行任意目录中的代码，对比查看在迭代过程中，由于代码的优化带来的性能提升效果。

- new_version_with_notes 目录： 这是本仓库的一个新版本，包含上述所有代码，里面的目录结构复刻了上述结构。区别在于给代码添加了注释，并且优化了一些细节。建议第一次使用的同学直接使用 **new_version_with_notes** 目录下的代码。


---
## 我是如何实现从零手写 resnet50 的

### 实现思路

#### 模型获取

使用 torchvision 从已经预训练好的模型中，将 resnet50 每一层的权值保存到仓库中，所保存的权值文件会在后续被加载进来，参与卷积、全连接、BN层的计算。

这里多说一些，在实际工业项目的模型部署中，神经网络的权值也是作为独立的数据被加载到GPU/CPU中完成计算的。

而很多实际模型的性能瓶颈会是在权值加载部分。为什么呢？我分析有几个原因：
- 受限于芯片内存的限制。导致无法将神经网络的所有权值全部一次加载，而多次加载带来的副作用便是会带来多余的IO操作，内存越小此问题越严重。
- 受限于芯片带宽的限制。在模型参数量日益增大的今天，GB 级别的带宽越来越显得吃力，而且在很多时候，IO 和计算无法真正在芯片上完全流水起来，尤其是在堆算力的时候，IO 就被凸显出来了。

- 在 model 目录下，运行以下脚本，即可将参数保存到 model/resnet50_weight 中。
```python
$ python3 resnet50_parser.py
```

#### 代码实现

在保存完权值后，利用 python / C++ 语言，分别实现 Conv2d, BatchNorm, Relu, AvgPool, MaxPool, FullyConnect(MatMul) 等核心函数。

按照 [resent50的网络结构](https://mp.weixin.qq.com/s/nIeu58iSy5-cTafkvFxdig), 将以上算法搭起来。
- 模型文件参考 [model/resnet50.onnx.png](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/model/resnet50.onnx.png) 和 [model/resnet50_structure.txt](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/model/resnet50_structure.txt) 
- 手工搭建 resnet50 的网络结构参考 [我手工搭建的模型, Python 版本](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/python/infer.py#L354)

#### 推理

代码实现完成后，意味着模型运行需要的基础算法和参数已经就位，下面读取一张本地图片，进行推理。
- 读取一只猫的图片，参考[获取图片](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/python/infer.py#L296)

读取完图片，开始推理，正确推理出来是一只猫，本项目第一阶段目标(准确性验证)即完成。

#### 优化

在基本功能实现完成后，开始着手进行性能优化。

性能优化属于神经网络中的一大重点，下面单分一章节来说明。

---
## 性能优化

### python 版本

这部分是 python 版本的性能优化，先看下本仓库如何使用 python 代码。

#### 怎么用 python 版本

1. resnet50 的核心算法和手搭网络是用基础的 python 语法写的，有些十分基础的操作调用 numpy 库。
2. 导入图片调用的 pillow 库，导入图片这种逻辑不属于**从零手写 resnet50 核心算法**的范畴，我也没时间去写类似的逻辑，直接用 pillow 库。
3. 安装依赖，主要是上面两个库的依赖（国内清华源比较快，可自己按需来选择），在 python 目录下，执行：

不使用清华源
```shell
$ pip3 install -r requirements.txt
```

使用清华源：
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 推理
- 在 python 目录下，运行以下命令，完成推理，你可以修改 my_infer.py 中的获取图片的逻辑，将图片替换成你自己的图片，看能否正确的识别出来。
```shell
$ python3 my_infer.py
```

由于 Python 版本也基本没有调用三方库，以 python 的语法来写卷积循环，其性能绝对差到惨不忍睹，实测发现用 python 版本推理一张图片十分缓慢，主要是循环太多(但是为了展示算法的内部实现)。

#### python 版本的一点优化

利用 np.dot(内积运算)代替卷积的乘累加循环。
- 优化 python 版本的算法实现：[优化版本](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/python/ops/conv2d.py#L73)

python 不调用三方库的话，很多优化点无法去做(比如指令集不好控制、内存不好控制)，下面还是重点优化C++版本。

---
### C++ 版本

这部分是 C++ 版本的性能优化，先看下本仓库如何使用 c++ 代码。

#### 怎么用 c++ 版本

本仓库的 C++ 代码已经合入了几次优化提交，每次都是在前一次优化的基础上做的进一步优化，优化记录可以通过 cpp 目录下的文件名很方便的看出来。

- [cpp/1st_origin](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/1st_origin) 目录下存放的是第一版的 C++ 代码
- [cpp/2nd_avx2](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/2nd_avx2) 目录下存放的是第二版的 C++ 代码，启用了 **avx2 指令集的优化，以及 -Ofast 编译选项**
- [cpp/3rd_preload](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/3rd_preload) 目录下存放的是第三版的 C++ 代码，利用类似内存池的方式，增加了**权值提前加载**的逻辑，仍保留了每一层结果输入输出的动态 malloc 过程。
- [cpp/4th_no_malloc](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/4th_no_malloc) 目录下存放是第四版优化的 c++ 代码，删除了所有动态内存申请的操作，大幅提高性能。
- [cpp/5th_codegen](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/5th_codegen) 目录下存放是第五版优化的 c++ 代码，利用 CodeGen 和 jit 编译技术生成核心计算逻辑。
- [cpp/6th_mul_thread](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/6th_mul_thread) 目录下存放是第六版优化的 c++ 代码，利用多线程优化卷积的运算，大幅提升性能。

#### 编译

每个版本的目录下文件是独立的，不存在依赖，如果你想看两个版本间的代码改动，可以使用源码比较工具来查看。

每个版本的目录下文件的编译过程是相同的。 如果你只有 windows 环境而没有 linux 环境，可以查看[不用虚拟机，10 分钟快速在 windows 下安装 linux 系统](https://mp.weixin.qq.com/s/iCdgm_vBnOFGa5b6yISKVA)这里快速安装一个linux系统，如果你购买了付费文章，会有更加详细的安装指导。

如果你有 linux 环境，并且对 linux 操作很熟悉，请直接往下看：

- C++ 版本编译依赖 opencv 库，用来进行图片的导入，功能与 python 版本的 pillow 类似，linux 环境下，执行以下命令安装 opencv 库：
```shell
$ sudo apt-get install libopencv-dev python3-opencv libopencv-contrib-dev
```
- cpp 目录下，运行 [compile.sh](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/1st_origin/compile.sh) 即可完成编译。
```shell
$ bash ./compile.sh
```
编译完成后，在当前目录下，生成名为 **resnet** 的可执行文件，直接执行该文件，会对仓库中保存的图片进行推理，并显示结果。
```shell
$ ./resnet
```

#### 初始版本一

目录为 [cpp/1st_origin](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/1st_origin)。

第一版没有考虑性能问题，仅仅是按照想法完成了功能，可想而知性能惨不忍睹，此版本性能数据：

| Average Latency | Average Throughput |
| ---- | ---- |
| 16923 ms | 0.059 fps |

性能数据和电脑性能有关，你可跑下试试，看看打印出来的 Lantency 是多少。

#### 优化版本二

目录为[cpp/2nd_avx2](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/2nd_avx2)。

第二版在第一版的基础上，将卷积算法中的**乘累加**的循环运算，利用向量指令集做了并行化加速，采用的向量指令集为 avx2，你可以通过以下命令查看你的 CPU 是否支持 avx2 指令集。

```shell
$ cat /proc/cpuinfo
```
在显示的信息中如果存在 avx2 便是支持该指令集。

此版本性能数据：

| Average Latency | Average Throughput |
| ---- | ---- |
| 4973 ms | 0.201 fps |

#### 优化版本三

目录为[cpp/3rd_preload](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/3rd_preload) 。

第三版在第二版的基础上，消除了运算推理过程中针对权值参数动态 malloc 的过程，改为在推理之前，利用 std::map 管理一个类内存池的结构，推理之前将所有的权值参数全部加载进来，这一步优化在实际模型部署中是有现实意义的。

模型参数的提前加载可以最大限度的减轻系统的IO压力，减少时延。

此版本性能数据：

| Average Latency | Average Throughput |
| ---- | ---- |
| 862 ms | 1.159 fps |

#### 优化版本四

目录为[cpp/4th_no_malloc](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/4th_no_malloc) 。

第四版在第三版的基础上，消除了运算推理过程中所有动态内存申请，以及与字符串相关的操作。

此版本性能数据：

| Average Latency | Average Throughput |
| ---- | ---- |
| 742 ms | 1.347 fps |

#### 优化版本五

目录为[cpp/5th_codegen](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/5th_codegen) 。

第五版在第四版的基础上，利用 CodeGen 技术生成核心计算逻辑，利用 jit 编译完成编译过程。

此版本性能数据：

| Average Latency | Average Throughput |
| ---- | ---- |
| 781 ms | 1.281 fps |

#### 优化版本六

目录为[cpp/6th_mul_thread](https://github.com/dongtuoc/cv_learning_resnet50/tree/main/new_version_with_notes/practice/cpp/6th_mul_thread)。

第六版在第五版的基础上，利用多线程来优化了卷积计算，对 co 维度进行了线程间的独立拆分，用满 CPU 线程数。

此版本性能数据：

| Average Latency | Average Throughput |
| ---- | ---- |
| 297 ms | 3.363 fps |

经过 6 个版本的优化，推理延时从 16923 ms 优化至 297 ms, 提升了近 60 倍的性能。推理一张图片已经感觉不到卡顿，算是不错的效果。

---
## 整体仓库依赖

1. 保存权值的依赖
- cd 到 model 目录，安装解析模型相关的依赖库。
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. python 推理依赖

- cd 到 python 目录，安装推理 resnet50 需要的依赖库，主要是 numpy 还有 Pillow 库，用来导入图片。
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---
## 其他 Contact me

- 本项目所有代码和相关文章，均为个人原创，未经同意，请勿随意转载至任何平台，更不可用于商业目的，我已委托相关维权人士对原创文章和代码进行监督。
- 如果你有其他相关事宜，欢迎和我交流(微信wechat：ddcsggcs，or email: dongtuoc@yeah.net)。

---
## 愿人人都可快速入门 AI 视觉。
