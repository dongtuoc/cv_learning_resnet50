# Resnet50Zero

---
## 项目介绍

### 本项目在做什么

本项目旨在进行 resnet50 的算法原理学习和实战,以及基于该网络的**性能优化**。

- 原理学习会输出一些包含解析 resnet50 中用到的算法和背景知识。
- 实战部分会用python/C++两种语言完成 resnet50 的网络手写。其中resnet50的所有核心算法和网络结构全部手写，不借用任何第三方库，由于是自己手写的算法和网络，因此会有很大的自由度进行性能优化，性能优化也就是本项目最后进行的部分，会持续很长时间，迭代很多个版本。
- 实战部分在完成手写算法的基础上，除了要保证网络精度可用（也就是任意给一张图片，经过预处理之后，Top5 可以正确的预测出图片）之外，更重要的我会关注在性能优化部分，这一点后面会有介绍。

#### 为什么要全部手写核心算法

目前很多教程或者课程，在教你手搭神经网络的时候，基本都是基于 torch 的 nn 模块或其他模块，用 nn.conv2d 完成卷积的计算。

对于不想深究算法和学习算法的同学，或者一些初学者而言，即使按照教程将神经网络搭建出来了，再进一步，将图片推理出来了，依旧是云里雾里，不知其原理，始终浮于表面，心里不踏实。
nn.conv2d 的调用将conv2d的实现封装起来了，看不到，很难学到里面的实现细节，跟别提如何在此基础上进行性能优化了。

于是我突发奇想，所有的代码全部自己写，便有了很大的自由度，可以十分方便的对神经网络进行优化（魔改），在确保精度的前提在，获取最好的性能。

这也是进行本项目的初衷。查看[从零手写resnet50开始啦](https://mp.weixin.qq.com/s/5ARwORt3qZPKPSOZdGbJdw)。

从2023年4月开始，陆陆续续调试了很多次，目前项目已经初见效果。代码早已完成，精度也很OK，现在正在着手进行性能优化，优化方法包括但不限于：指令集优化、并行计算、内存调优(复用、权值前提驻留)等等。


### 你可以学到什么

通过本项目，你可以深入理解 resnet50 中用到的所有算法原型、算法的背景和原理、resent50 的思想、resnet50 的网络结构、以及常见的神经网络优化方法，并且你可以参考项目中给出的代码，真正运行一个 resnet50 神经网络，完成一张或多张图片的推理。

在关键的地方我都会给出文字详解，如果你把项目涉及的链接文章都阅读一遍，我觉得关于 resnet50 的问题，即使你是一个小白，也可以出师了。

最后在阅读了文章之后，跟着项目中的代码进行练习，将代码解析我也会抽时间写一写。

在学习该项目的过程中，有任何疑问都可以随时联系我（微信号: ddcsggcs）。


***
## 项目所涉及文章列表

文章列表中展示的文章链接，皆为我的原创文章。分两个部分：**原理解析和项目实战**。

**原理解析**部分，是我对 resnet50 这一神经网络，用通俗易懂的语言，写的算法和原理的拆解，有助于帮助入门的小伙伴快速了解算法。

**项目实战**部分，是我在对本项目写代码、调试过程中，遇到的一些问题和总结，可以看作是项目完成的过程记录。

### 原理解析

- [1 从像素说起](https://mp.weixin.qq.com/s/wKN9pBy6-pwH90oozoT2Cg)
- [2 图像的色彩空间](https://mp.weixin.qq.com/s/WKmHgF0_ZEAzol5sMVWwkw)
- [3 初识卷积](https://mp.weixin.qq.com/s/TojVegd5nadoa6n3TkhDkg)
- [4 卷积的核心，特征提取](https://mp.weixin.qq.com/s/wJiWafpIioe2h60bphw4gA)
- [5 残差结构](https://mp.weixin.qq.com/s/2ezkSTYhXfS480VKm0XSbA)
- [6 resent50的网络结构](https://mp.weixin.qq.com/s/nIeu58iSy5-cTafkvFxdig)
- [7 激活函数](https://mp.weixin.qq.com/s/aUfkmEr3MSwCL9k6QwNXfw)
- [8 池化层](https://mp.weixin.qq.com/s/9QIMSgOKUoOWI80JLKcMvA)
- [9 全连接是什么](https://mp.weixin.qq.com/s/sotJuoFis4t6PWfDjNSZoQ)
- [10 彻底搞懂 softmax](https://mp.weixin.qq.com/s/cTVhOFwHxgLfWEPup3tcCA)
- [11 总结篇：1.8w字趣解resnet50的算法原理](https://mp.weixin.qq.com/s/pA86udkaFzCogi2Qw8vBEA)

### 项目实战

- [1 准备从零开始手写 resnet50 了](https://mp.weixin.qq.com/s/5ARwORt3qZPKPSOZdGbJdw)
- [2 权值参数保存](https://mp.weixin.qq.com/s/ovplStVgqk25jkwSi4XYBQ)
- [3 手写龟速卷积](https://mp.weixin.qq.com/s/O9qjjmpuiaIIVpejhYBS9A)
- [4 利用 torch来debug，识别出了萨摩耶](https://mp.weixin.qq.com/s/YvUSh5UpUxq0eCIvsybDBQ)
- [5 我完全手写的算法和网络，识别出了虎猫](https://mp.weixin.qq.com/s/jHlTWt-pH4glmtTP6pcKFw)
- [6 大提速，分分钟识别“十二生肖”](https://mp.weixin.qq.com/s/8ym0bRR-miASQmbZmfvuag)
- 未完待续。


---
## 我是如何实现本项目的

### 项目实现思路

#### 模型获取

使用 torchvision 从已经预训练好的模型中，将 resnet50 每一层的权值保存到仓库中，所保存的权值文件会在后续被加载进来，参与卷积、全连接、BN层的计算。

在实际工业项目的模型部署中，神经网络的权值也是作为独立的数据被加载到GPU/CPU中完成计算的。

而很多实际模型的性能瓶颈会是在权值加载部分。为什么呢？我分析有几个原因：
- 受限于芯片内存的限制。导致无法将神经网络的所有权值全部一次加载，而多次加载带来的副作用便是会带来多余的IO操作，内存越小此问题越严重。
- 受限于芯片带宽的限制。在模型参数量日益增大的今天，GB 级别的带宽越来越显得吃力，而且在很多时候，IO 和计算无法真正在芯片上完全流水起来，尤其是在堆算力的时候，IO 就被凸显出来了。

关于权值从模型中保存的实现：

- 文章参考：[权值参数保存](https://mp.weixin.qq.com/s/ovplStVgqk25jkwSi4XYBQ)
- 在仓库 model 目录下，运行以下脚本，即可将参数保存到 model/resnet50_weight 中。
```python
$ python3 resnet50_parser.py
```

#### 代码实现

在保存完权值后，利用 python / C++ 语言，分别实现 Conv2d, BatchNorm, Relu, AvgPool, MaxPool, FullyConnect(MatMul) 等核心函数。

按照 [resent50的网络结构](https://mp.weixin.qq.com/s/nIeu58iSy5-cTafkvFxdig), 将以上算法搭起来。
- 模型文件参考 [model/resnet50.onnx.png](https://github.com/dongtuoc/resnet50_zero/blob/master/model/resnet50.onnx.png) 和 [model/resnet50_structure.txt](https://github.com/dongtuoc/resnet50_zero/blob/master/model/resnet50_structure.txt) 
- 手工搭建 resnet50 的网络结构参考 [我手工搭建的模型, Python 版本](https://github.com/dongtuoc/resnet50_zero/blob/master/python/my_infer.py#L273)

#### 推理

代码实现完成后，意味着模型运行需要的基础算法和参数已经就位，下面读取一张本地图片，进行推理。
- 读取一只猫的图片，参考[获取图片](https://github.com/dongtuoc/resnet50_zero/blob/master/python/my_infer.py#L241)

读取完图片，开始推理，参考[python/my_infer.py](https://github.com/dongtuoc/resnet50_zero/blob/master/python/my_infer.py)文件。正确推理出来是一只猫，本项目第一阶段（精度验证）即完成——[我完全手写的算法和网络，识别出了虎猫](https://mp.weixin.qq.com/s/jHlTWt-pH4glmtTP6pcKFw)

#### 优化

在功能实现完成后，开始性能优化。

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

由于 Python 版本也基本没有调用三方库，以 python 的语法来写卷积循环，其性能绝对差到惨不忍睹，实测发现用 python 版本推理一张图片在分钟级别，主要是循环太多（为了展示算法的内部实现）。

#### 优化1

利用 np.dot（内积运算）代替卷积的乘累加循环。
- 优化 python 版本的算法实现：[大提速，分分钟识别“十二生肖”](https://mp.weixin.qq.com/s/8ym0bRR-miASQmbZmfvuag)

python 不调用三方库的话，很多优化点无法去做(比如指令集不好控制、内存不好控制)，下面还是重点优化C++版本。

---
### C++ 版本

这部分是 C++ 版本的性能优化，先看下本仓库如何使用 c++ 代码。

#### 怎么用 c++ 版本

本仓库的 C++ 代码已经合入了几次优化提交，每次都是在前一次优化的基础上做的进一步优化，优化记录可以通过 cpp 目录下的文件名很方便的看出来。

- [cpp/1st_version](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/1st_version) 目录下存放的是第一版的 C++ 代码
- [cpp/2nd_version_avx2](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/2nd_version_avx2) 目录下存放的是第二版的 C++ 代码，启用了 **avx2 指令集的优化，以及 -Ofast 编译选项**
- [cpp/3rd_version_avx2_preload](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/3rd_version_avx2_preload) 目录下存放的是第三版的 C++ 代码，利用累死内存池的方式，增加了**权值提前加载**的逻辑，仍保留了每一层结果输入输出的动态 malloc 过程。

#### 编译

每个版本的目录下文件是独立的，不存在依赖，如果你想看两个版本间的代码改动，可以使用源码比较工具来查看。

每个版本的目录下文件的编译过程是相同的，如下。

- C++版本编译依赖 opencv 库，用来进行图片的导入，功能与 python 版本的 pillow 类似，linux 环境下，执行以下命令安装 opencv 库：
```shell
$ sudo apt-get install libopencv-dev python3-opencv libopencv-contrib-dev
```
- cpp 目录下，运行 [build.sh](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/1st_version/build.sh) 即可完成编译。
```shell
$ bash ./build.sh
```
编译完成后，在当前目录下，生成名为 **resnet** 的可执行文件，直接执行该文件，会对仓库中保存的图片进行推理，并显示结果。
```shell
$ ./resnet
```

#### 初始版本一

目录为[cpp/1st_version](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/1st_version)。

第一版没有考虑性能问题，仅仅是按照想法完成了功能，可想而知性能惨不忍睹，基本上10000 ms（10几秒）推理一张图片。

和电脑性能优化，你可跑下试试，看看打印出来的耗时是多少。

#### 优化版本二

目录为[cpp/2nd_version_avx2](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/2nd_version_avx2)。

第二版在第一版的基础上，将卷积算法中的**乘累加**的循环运算，利用向量指令集做了并行化加速，采用的向量指令集为 avx2，你可以通过一下命令查看你的 CPU 是否支持 avx2 指令集。
```shell
$ cat /proc/cpuinfo
```
在显示的信息中如果存在 avx2 便是支持该指令集。

重点优化修改查看：[向量指令集的替换](https://github.com/dongtuoc/resnet50_zero/blob/main/cpp/2nd_version_avx2/resnet_avx2.h#L72)

优化完之后，一张图片的[延时](https://mp.weixin.qq.com/s/-Dj9Ck7Ii3rtO2fa3rXIAw)约为 4000 ms，已比上一版本有了很大的提升。

#### 优化版本三

目录为[cpp/3rd_version_avx2_preload](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/3rd_version_avx2_preload)。

第三版在第二版的基础上，消除了运算推理过程中针对权值参数动态malloc 的过程，改为在推理之前，利用 std::map 管理一个类内存池的结构，推理之前将所有的权值参数全部加载进来，这一步优化在实际模型部署中是有现实意义的。

模型参数的提前加载可以最大限度的减轻系统的IO压力，减少时延。

重点优化修改查看：[PreLoad](https://github.com/dongtuoc/resnet50_zero/blob/main/cpp/3rd_version_avx2_reload/resnet_avx2_preload.h#L263)，可查看[实现](https://github.com/dongtuoc/resnet50_zero/blob/main/cpp/3rd_version_avx2_reload/resnet_avx2_preload.h)中所有带有 **PRE_LOAD_PARAM** 的地方。

这一版加入了平均时延(Latency)的统计，以及吞吐量(Throughput)的计算。

优化完之后，平均一张图片的[延时](https://mp.weixin.qq.com/s/-Dj9Ck7Ii3rtO2fa3rXIAw)约为 865.5 ms，吞吐 1.15 fps。

---
## 开发日志

### 2023-4-13

由于手写的卷积算法，没有用到任何第三方模块加速计算，因此，一层卷积的计算时间较长，需要你能忍耐这个时间，忍耐的办法就是等。

我测试了一下第一层的卷积，大概1min左右计算完成。

如此算下来，整个网络有50个卷积层，还有其他计算，大概1个小时能完成那张猫的推理。

不过这都不是问题，我们在学习整个思路。等真的能花1个小时把猫推理出来之后，下一步的计划就是：怎么加速推理。

争取秒级的推理速度。

### 2023-4-14

conv, bn, relu, add, maxpool, avgpool 的算法都已经开发完成，按照模型结构搭建起来了模型。

推理了一下，耗时40分钟，主要集中在卷积的循环计算上。这个后续优化。

分类结果还需要查找资料。

### 2023-4-20

分类结果已经从网上下载下来了，放在了仓库中：imagenet_classes.txt， 共1000个分类。

另外，自己搭的神经网络推理完，发现识别错误，开始debug。

发现了一个地方：torch 中的模型权值是按照NCHW摆放的，而我手写的算法全部按照NHWC摆放的，因此在dump权值的时候，添加一个 transpose 操作，此bug已修复。

修复完成后，使用 torch 快速搭建了一个 resnet50模型，用用他来推理 pics 目录下的两种图片，均能正确识别出是虎猫和萨摩耶。

而我手写的模型识别不对，说明中间的计算环节有错误。于是开始debug。

逐层将 torch 的计算结果打印出来，和我手写的计算结果做对比，发现 resnet50.conv1, bn1, relu 的结果都能对的上。

第一个layer1，conv1->b1->conv2->bn1->conv3->bn3->relu的结果也能对的上, 其中单独的下采样 conv->bn 的结果也是对的。

但layer1最终的输出和我的layer1最终的输出存在差异，结果对不上，说明在处理残差结构的时候出了问题。

这个问题后面继续查找。先记录一下。

### 2023-4-23

基本逻辑已经调通，可以出 v0.1版本了。

全部使用自己手写的算法和网络，已经try通整个流程，正确的识别出一只猫了。[出猫了](https://mp.weixin.qq.com/s?__biz=MzAwOTc2NDU3OQ==&mid=2649036515&idx=1&sn=43906ed7ec11641361bbf099c1f6e1bc&chksm=834b1d6fb43c9479567f905b269519a38641927b7f955cd12002b91c174c50954dd1e9ad39aa&token=775743764&lang=zh_CN#rd)

备份跑通的v0.1版本，见[v0.1 python版本代码备份](https://github.com/dongtuoc/resnet50_zero/blob/master/python/v0.1_debug_myInfer.bk.py)。

### 2023-5-4

最近没有更新文章，readme 也没更新，但是项目已经有了更大的进展。
1.  C++版本已经实现完成，并且功能正常，可以正确的推理出猫和萨摩耶了。
2.  C++版本性能比python版本快很多，目前的最佳性能，推理一张图片耗时 6000ms。主要包括：加载所有层的权值、预处理、推理计算的耗时。
3. 目前启用了两种优化方法：
- 向量指令：使用 avx2 指令，替代标量的乘法和加法，主要优化卷积计算，对比 [resnet.h](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/resnet.h)和[resent_avx2.h](https://github.com/dongtuoc/resnet50_zero/blob/master/cpp/resnet_avx2.h)就可以看到优化差异。类似与 python 优化中使用 np.dot 优化乘累加，查看 [np.dot 优化乘累加](https://github.com/dongtuoc/resnet50_zero/blob/master/python/my_infer.py#L28)。
- -Ofast 优化：在编译脚本中，启用 g++ 的 -Ofast 优化，其优化的性能比 -O3 还要好。

### 2023-11-03

终于有时间再来维护一下这个仓库了，看到有一些小伙伴星标了本仓库，感谢，如果您在看，也请您动动小手手星标一下吧。

今天利用类内存池的东西，将权值做了提前加载，推理时延终于降到了1s以下，平均800ms。

但是，还有不规范的地方在于，每一层输入输出的内存还是采用动态申请的方式，这一点是不能忍的，因为在实际项目中，整个推理过程，是不允许和操作系统做这种 malloc 行为，所以下一步会把所有的推理路径上的 malloc 动作全部干掉，性能应该还能再上一大截。


---
## 仓库结构

1. [model](https://github.com/dongtuoc/resnet50_zero/tree/master/model) 模型目录
- [resnet50_weight](https://github.com/dongtuoc/resnet50_zero/tree/master/model/resnet50_weight) 保存的权值文件
- [requirements.txt](https://github.com/dongtuoc/resnet50_zero/blob/master/model/requirements.txt) 保存权值需要的依赖库
- [resnet50.onnx.png](https://github.com/dongtuoc/resnet50_zero/blob/master/model/resnet50.onnx.png) resnet50 的可视化网络结构
- [resnet50_structure.txt](https://github.com/dongtuoc/resnet50_zero/blob/master/model/resnet50_structure.txt) resnet50 的结构
- [resnet50_parser.py](https://github.com/dongtuoc/resnet50_zero/blob/master/model/resnet50_parser.py) 解析 resnet50，并且将每一层的权值参数保存下来的脚本

---
## 整体仓库依赖

1. 保存权值的依赖
- cd 到 model 目录，安装解析模型相关的依赖库。
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. python 推理依赖

- cd 到 python 目录，安装推理resnet50需要的依赖库，主要是 numpy 还有 Pillow 库，用来导入图片。
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---
## 其他

- 如果你感兴趣，可以联系我，一起维护本仓库，你可以按照自己的想法来优化一些算法和性能，只要有效果，都十分欢迎。
- 本项目所有代码和所列的所有的文章，均为我个人原创，未经同意，请勿随意转载至任何平台，更不可用于商业目的，我已委托相关维权人士对我的原创文章和代码进行监督。
- 如果你有其他相关事宜，欢迎和我交流。

---
## 愿 resnet50 这一神经网络，对所有人都不是难题
