# Faster RCNN Cpp

An implementation of Faster R-CNN with VGG16 backbone (https://arxiv.org/pdf/1506.01497.pdf) in libtorch.

## Features

+ Supports PASCAL VOC 2007 and MS COCO 2017 datasets
+ Supports VGG16 backbone (from official torchvision model)
+ Supports ROI Pool and ROI Align modes
+ Supports PASCAL VOC 2007 and PASCAL VOC 2012 evaluation

## Requirements

+ libtorch
+ torchvision
+ OpenCV
+ Boost

## Build

```bash

cd faster-rcnn-cpp

cmake -S . -B build -DCMAKE_PREFIX_PATH=path_to_libtorch -DCMAKE_PREFIX_PATH=path_to_torchvision -DCMAKE_PREFIX_PATH=path_to_opencv -DCMAKE_PREFIX_PATH=path_to_boost

cmake --build build

```

## Todo

+ 数据集变换采用 libtorch 的 *Transform* 接口
+ 优化评价方法，测试后自动进行评价

## Benchmark

### PASCAL VOC 2007 (Train/Test: 07trainval/07test, non-difficult)

batch size: 1, lr: 0.00125, decay epoch: 9 12, total epoch: 12

| COCO Metric | AP | AP50 |
| ---------- | -------- | ---------- |
| ResNet50_FPN (this)  |  **0.439**   | 0.767 |
| ResNet50_FPN (mmdet) | 0.437 | **0.769** |
| ResNet50_FPN (base)  | 0.438 | 0.768|

|    VOC 2007 Metric        | mAP |
| ---------- | -------- |
|  ResNet50_FPN (this)  |  0.7581   |
|  VGG16:VGG-feature + 1024*1024 + random-flip (this)  |  0.6708   |
|  VGG16:VGG-feature + 4096*4096 + random-flip (this)  | 0.6751   |
|  VGG16:VGG-feature + 4096*4096 + append-flip (this)  | 0.6778(12) **0.6798**(10) |
|  VGG16:VGG-feature + VGG-classifier + without-flip (this)  | 0.6478   |
|  VGG16:VGG-feature + VGG-classifier + random-flip (this)  | 0.6650   |
|  VGG16:VGG-feature + VGG-classifier + append-flip (this)  | 0.6664(12) 0.6685(10) |
|  VGG16:VGG-feature + VGG-classifier + without-flip (Pytorch)  |  0.674   |
|[Original Paper](https://arxiv.org/abs/1506.01497)|0.699|
|  [VGG16:VGG-feature + VGG-classifier + append-flip (Pytorch)](https://github.com/jwyang/faster-rcnn.pytorch)  |  **0.701**   |
## Acknowledgement

This project is based on the following project:

+ https://github.com/thisisi3/libtorch-faster-rcnn
