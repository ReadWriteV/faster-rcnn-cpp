# Faster RCNN Cpp

Faster RCNN、DA Faster RCNN的libtorch实现。

## Feature

+ VOC 和 COCO 两种常见数据集实现
+ VOC 评价方法实现
+ VGG 系列和 ResNet 系列特征提取网络的实现

## Todo

+ 数据集变换采用 libtorch 的接口 *Transform*
+ boost::property_tree::ptree 实现 *get&lt;std::vector&lt;T>>*

## Benchmark

PASCAL VOC 2007 (Train/Test: 07trainval/07test, non-difficult, ROI Align)

| COCO Metric | AP | AP50 |
| ---------- | -------- | ---------- |
|  this (ResNet50_FPN)  |  0.439   | 0.767 |
| mmdet (ResNet50_FPN) | 0.437 | 0.769 |
|base (ResNet50_FPN) |0.438 | 0.768|
|best (ResNet50_FPN) |0.439 | 0.771|

|    VOC Metric        | AP |
| ---------- | -------- |
|  this (ResNet50_FPN)  |  0.7545   |
|  best (ResNet50_FPN)  |  0.7606   |
|  this (VGG16)  |  ?   |
|  Pytorch (VGG16) |  0.7010   |

### faster rcnn Pytorch

VGG16 6 epoch

```
AP for aeroplane = 0.7392
AP for bicycle = 0.7772
AP for bird = 0.6699
AP for boat = 0.5580
AP for bottle = 0.5223
AP for bus = 0.7768
AP for car = 0.8504
AP for cat = 0.8243
AP for chair = 0.4959
AP for cow = 0.7750
AP for diningtable = 0.6306
AP for dog = 0.7800
AP for horse = 0.8317
AP for motorbike = 0.7476
AP for person = 0.7715
AP for pottedplant = 0.4468
AP for sheep = 0.7059
AP for sofa = 0.6388
AP for train = 0.7236
AP for tvmonitor = 0.7538
Mean AP = 0.7010
```

# Acknol