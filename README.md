# Faster RCNN Cpp

Faster RCNN 的 libtorch 实现。

## Feature

+ VOC 和 COCO 两种常见数据集实现
+ VOC 评价方法实现

## Todo

+ 数据集变换采用 libtorch 的 *Transform* 接口

## Benchmark

### PASCAL VOC 2007 (Train/Test: 07trainval/07test, non-difficult)

batch size: 1, lr: 0.00125, decay epoch: 9 12, total epoch: 12

batch size: 1, lr: 1e-3, decay epoch: 5, total epoch: 6 

| COCO Metric | AP | AP50 |
| ---------- | -------- | ---------- |
|  ResNet50_FPN (this)  |  0.439   | 0.767 |
| ResNet50_FPN (mmdet) | 0.437 | 0.769 |
|  ResNet50_FPN (base)  |0.438 | 0.768|

|    VOC Metric        | mAP |
| ---------- | -------- |
|  ResNet50_FPN (this)  |  0.7581   |
|  VGG16:VGG-feature + 1024*1024 + random-flip (this)  |  0.6708   |
|  VGG16:VGG-feature + 4096*4096 + random-flip (this)  | 0.6751   |
|  VGG16:VGG-feature + VGG-classifier + random-flip (this)  | 0.6650   |
|  VGG16:VGG-feature + VGG-classifier + without-flip (this)  | 0.6478   |
|  VGG16:VGG-feature + ?classifier + append-flip (this)  | 0.6796   |
|  VGG16:VGG-feature + VGG-classifier + without-flip (Pytorch)  |  0.674   |
|  VGG16:VGG-feature + VGG-classifier + append-flip (Pytorch)  |  0.7010   |

### Foggy Cityscapes (Train/Test: train/val)

batch size: 1, lr: 0.00125, decay epoch: 9 12, total epoch: 12

batch size: 1, lr: 2e-3, decay epoch: 6, total epoch: 12 

|    VOC Metric        | mAP |
| ---------- | -------- |
|  ResNet50_FPN (this)  |  0.4387   |
|  VGG16:VGG-feature + 1024*1024 + random-flip (this)  |  0.2792   |
|  VGG16:VGG-feature + 4096*4096 + random-flip (this)  | 0.2919   |
|  VGG16:VGG-feature + VGG-classifier + random-flip (this)  | ?   |
| VGG16:VGG-feature + 4096*4096 + rpn_conv1_512 + random-flip (this) |  0.2844  |
|  VGG16:VGG-feature + ?classifier + append-flip (this)  | ?   |
|  VGG16:VGG-feature + VGG-classifier + append-flip (Pytorch)  |  0.4847   |

## Acknowledgement

This project is based on the following project:

+ https://github.com/thisisi3/libtorch-faster-rcnn
