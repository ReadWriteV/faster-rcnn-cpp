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

## Benchmark

### PASCAL VOC 2007 (Train/Test: 07trainval/07test, non-difficult)

batch size: 1, lr: 0.00125, decay epoch: 9 12, total epoch: 12

| COCO Metric | AP | AP50 |
| ---------- | -------- | ---------- |
| ResNet50_FPN (this)  |  **0.439**   | 0.767 |
| ResNet50_FPN (mmdet) | 0.437 | **0.769** |
| [ResNet50_FPN](https://github.com/thisisi3/libtorch-faster-rcnn)  | 0.438 | 0.768|

|    VOC 2007 Metric        | mAP |
| ---------- | -------- |
|  VGG16 (this)  | 0.6798 |
|[Original Paper](https://arxiv.org/abs/1506.01497)|0.699|
|  [VGG16 (Pytorch)](https://github.com/jwyang/faster-rcnn.pytorch)  |  **0.701**   |

## Todo

+ Implement the dataset transform with the interface of libtorch
+ Integrate evaluation into dataset and automatically run evaluation after test

## Acknowledgement

This project is based on the following project:

+ https://github.com/thisisi3/libtorch-faster-rcnn
