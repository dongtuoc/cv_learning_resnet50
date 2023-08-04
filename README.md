# Resnet50Zero

***
## 项目介绍

### 本项目在做什么

本项目旨在进行 resnet50 的算法原理学习和实战。

本项目会从零开始，不调用任何第三方库，手写 resnet50 这一神经网络中的所有算法，并且按照 resnet50 的网络结构搭建起来，最终完成一张图片的推理。

**不调用任何第三方库**，指的是核心算法和神经网络的搭建不会调用任何第三库。

目前很多教程手搭神经网络，基本都是基于 torch 的 nn 模块，用 nn.conv2d 就完成卷积的计算，但是卷积算法是如何实现的呢？被封装起来了，看不到，不能真正学到里面的实现细节，但是本项目，会把所有核心算法全部用 python 和 c++ 都手写一遍，然后手动搭建网络结构。

这也是进行本项目的初衷。查看[从零手写resnet50开始啦](https://mp.weixin.qq.com/s/5ARwORt3qZPKPSOZdGbJdw)。

### 你可以学到什么

通过本项目，你可以深入理解 resnet50 中用到的所有算法原型、算法的背景和原理、resent50 的思想、resnet50 的网络结构，并且你可以参考项目中给出的代码，真正运行一个 resnet50 神经网络，完成一张或多张图片的推理。

如果你把项目涉及的链接文章都阅读一遍，相信我，把这个项目写在简历上，关于 resnet50 的问题，你一点不会害怕。

大部分文章都是我自己写的，我基本都是用通俗易懂的语言来解析所有算法。

这个文档中，在适当的位置，我会给出文章链接，在下一节也会给出所有文章链接列表。

在阅读了文章之后，可以跟着项目中的代码进行练习，后面我会抽时间将代码解析写一写。


***
## 项目所涉及文章列表


文章列表中展示的文章链接，皆为我的原创文章。分两个部分：原理解析和项目实战。

原理解析部分，是我对 resnet50 这一神经网络，用通俗易懂的语言，写的算法和原理的拆解，有助于帮助入门的小伙伴快速了解算法。

项目实战部分，是我在对本项目写代码、调试过程中，遇到的一些问题和总结，可以看作是项目完成的过程记录。



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
- [11 总结篇：1.8w字解析resnet50的算法原理](https://mp.weixin.qq.com/s/pA86udkaFzCogi2Qw8vBEA)

### 项目实战


- [1 准备从零开始手写 resnet50 了](https://mp.weixin.qq.com/s/5ARwORt3qZPKPSOZdGbJdw)
- [2 权值参数保存](https://mp.weixin.qq.com/s/ovplStVgqk25jkwSi4XYBQ)
- [3 手写龟速卷积](https://mp.weixin.qq.com/s/O9qjjmpuiaIIVpejhYBS9A)
- [4 利用 torch来debug，识别出了萨摩耶](https://mp.weixin.qq.com/s/YvUSh5UpUxq0eCIvsybDBQ)
- [5 我完全手写的算法和网络，识别出了虎猫](https://mp.weixin.qq.com/s/jHlTWt-pH4glmtTP6pcKFw)
- [6 大提速，分分钟识别“十二生肖”](https://mp.weixin.qq.com/s/8ym0bRR-miASQmbZmfvuag)

未完待续。


***
## 我是如何实现本项目的

本项目实现的大致思路为：

1. 使用 torchvision 从已经预训练好的模型中，将 resnet50 每一层的权值保存到仓库中，权值文件后续参与卷积、全连接、BN层的计算。

- 文章参考：[权值参数保存](https://mp.weixin.qq.com/s/ovplStVgqk25jkwSi4XYBQ)

- 在仓库 model 目录下，运行以下脚本，即可将参数保存到 model/resnet50_weight 中。
```python
python3 resnet50_parser.py
```

2. 在保存完权值后。利用 python / C++ 语言，不借助第三方库，实现 my_conv2d, my_bn, my_relu, my_add, my_pool, my_fc 等核心函数。

3. 按照 [resent50的网络结构](https://mp.weixin.qq.com/s/nIeu58iSy5-cTafkvFxdig), 将以上算法（也就是每一层），手工搭建起来。

- 模型文件参考 [model/resnet50.onnx.png](https://gitee.com/iwaihou/resnet50-zero/blob/master/model/resnet50.onnx.png) 和 [model/resnet50_structure.txt](https://gitee.com/iwaihou/resnet50-zero/blob/master/model/resnet50_structure.txt) 
- 手工搭建 resnet50 的网络结构参考 [我手工搭建的模型](https://gitee.com/iwaihou/resnet50-zero/blob/master/python/my_infer.py#L273)

4. 以上2,3步完成后，意味着模型运行需要的基础算法和参数已经就位，下面读取一张本地图片，进行推理。
- 读取一只猫的图片，参考[获取图片](https://gitee.com/iwaihou/resnet50-zero/blob/master/python/my_infer.py#L241)

5. 读取完图片，就开始推理，参考[python/my_infer.py](https://gitee.com/iwaihou/resnet50-zero/blob/master/python/my_infer.py)文件。正确推理出来是一只猫，本项目第一阶段（功能）即完成——[我完全手写的算法和网络，识别出了虎猫](https://mp.weixin.qq.com/s/jHlTWt-pH4glmtTP6pcKFw)

6. 在功能实现完成后，下面开始性能优化：
- 优化 python 版本的算法实现：[大提速，分分钟识别“十二生肖”](https://mp.weixin.qq.com/s/8ym0bRR-miASQmbZmfvuag)
- 实现 C++ 版本：参考[C++ 的实现](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/main.cc)

***

## 如何使用 python 版本和 C++ 版本

本项目实现了两个版本，python 版本和 C++ 版本，两者实现的思路大致相同，主要是为了方便不同技术栈的同学学习和交流。

如果你对 python 熟悉，可以查看根目录下的 python 目录中的文件。

如果你对 C++ 熟悉，可以查看根目录下的 cpp 目录中的文件。

### 怎么用 python 版本

1. resnet50 的核心算法和手搭网络是用基础的 python 语言写的，有些十分基础的操作调用 numpy 库。
2. 导入图片调用的 pillow 库，导入图片这种逻辑不属于从零手写 resnet50 核心算法的范畴，我也没时间去写类似的逻辑，直接用 pillow 库即可。
3. 安装依赖，主要是上面两个库的依赖，在 python 目录下，执行
```shell
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
4. 推理
- 在 python 目录下，运行以下命令，完成推理。你可以修改 my_infer.py 中的获取图片的逻辑，将图片替换成你自己的图片，看能否正确的识别出来。
```shell
$ python3 my_infer.py
```

### 怎么用 c++ 版本

1. c++ 版本的核心算法在 [cpp/resnet.h](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/resnet.h) 中，与之相对应的，[cpp/resnet_avx2.h](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/resnet_avx2.h)文件是使用 avx2 指令对一些向量乘累加操作做的**向量优化**，感兴趣的同学可以看看。
- 如果你的电脑不支持 avx2 指令，将 [USE_AVX2_INST](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/main.cc#L15)这个宏置为0即可。
- 如何查看电脑是否支持 avx2 指令?在 linux 环境下运行以下命令，在显示的信息中查看是否有 avx2 指令集即可。
```shell
$ cat /proc/cpuinfo
```

2. 编译
- C++版本编译依赖 opencv 库，用来进行图片的导入，功能与 python 版本的 pillow 类似。执行以下命令安装 opencv 库：
```shell
$ sudo apt-get install libopencv-dev python3-opencv libopencv-contrib-dev
```
- cpp 目录下，运行 [build.sh](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/build.sh) 即可完成编译，在当前目录下，生成 a.out 的可执行文件。
```shell
$ ./build.sh
```

3. 推理

- 编译完成后，直接运行 a.out 即可完成推理。你可以修改 main.cc 中的 [getFileName()](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/main.cc#L23)函数来自定义需要出的图片。
```shell
$ a.out
```
***
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

备份跑通的v0.1版本，见[v0.1 python版本代码备份](https://gitee.com/iwaihou/resnet50-zero/blob/master/python/v0.1_debug_myInfer.bk.py)。

### 2023-5-4

最近没有更新文章，readme 也没更新，但是项目已经有了更大的进展。
1.  C++版本已经实现完成，并且功能正常，可以正确的推理出猫和萨摩耶了。
2.  C++版本性能比python版本快很多，目前的最佳性能，推理一张图片耗时 6000ms。主要包括：加载所有层的权值、预处理、推理计算的耗时。
3. 目前启用了两种优化方法：
- 向量指令：使用 avx2 指令，替代标量的乘法和加法，主要优化卷积计算，对比 [resnet.h](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/resnet.h)和[resent_avx2.h](https://gitee.com/iwaihou/resnet50-zero/blob/master/cpp/resnet_avx2.h)就可以看到优化差异。类似与 python 优化中使用 np.dot 优化乘累加，查看 [np.dot 优化乘累加](https://gitee.com/iwaihou/resnet50-zero/blob/master/python/my_infer.py#L28)。
- -Ofast 优化：在编译脚本中，启用 g++ 的 -Ofast 优化，其优化的性能比 -O3 还要好。


## 仓库结构


1. [model](https://gitee.com/iwaihou/resnet50-zero/tree/master/model) 模型目录
- [resnet50_weight](https://gitee.com/iwaihou/resnet50-zero/tree/master/model/resnet50_weight) 保存的权值文件
- [requirements.txt](https://gitee.com/iwaihou/resnet50-zero/blob/master/model/requirements.txt) 保存权值需要的依赖库
- [resnet50.onnx.png](https://gitee.com/iwaihou/resnet50-zero/blob/master/model/resnet50.onnx.png) resnet50 的可视化网络结构
- [resnet50_structure.txt](https://gitee.com/iwaihou/resnet50-zero/blob/master/model/resnet50_structure.txt) resnet50 的结构
- [resnet50_parser.py](https://gitee.com/iwaihou/resnet50-zero/blob/master/model/resnet50_parser.py) 解析 resnet50，并且将每一层的权值参数保存下来的脚本


## 仓库依赖

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

- 如果你感兴趣，可以联系我，一起维护本仓库。
- 我个人感觉，本仓库的内容，包括原理解析和实战代码训练，对于了解整个 resnet50 是足够了。
- 本项目所有代码和所列的所有的文章，均为我个人原创，未经同意，请勿随意转载至任何平台，更不可用于商业目的，我已委托相关维权人士对我的原创文章和代码进行监督。
- 如果你有其他相关事宜，欢迎和我交流。

---
## 愿 resnet50 这一神经网络，对所有人都不是难题
