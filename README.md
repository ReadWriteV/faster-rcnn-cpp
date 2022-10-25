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

VGG16 6 epoch without flipped

```
AP for aeroplane = 0.6669
AP for bicycle = 0.7746
AP for bird = 0.6803
AP for boat = 0.5070
AP for bottle = 0.5115
AP for bus = 0.7282
AP for car = 0.8189
AP for cat = 0.8013
AP for chair = 0.4790
AP for cow = 0.7246
AP for diningtable = 0.5721
AP for dog = 0.7391
AP for horse = 0.8211
AP for motorbike = 0.7450
AP for person = 0.7603
AP for pottedplant = 0.4085
AP for sheep = 0.6820
AP for sofa = 0.6429
AP for train = 0.6952
AP for tvmonitor = 0.7293
Mean AP = 0.674
```

VGG16 6 epoch with flipped

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

Result of VGG-feature + VGG-classify + old-RPN

```
aeroplane   : AP = 0.6190, PRE = 0.2212, REC = 0.7754
bicycle     : AP = 0.6978, PRE = 0.2364, REC = 0.8783
bird        : AP = 0.6119, PRE = 0.2516, REC = 0.7495
boat        : AP = 0.4434, PRE = 0.1176, REC = 0.7871
bottle      : AP = 0.3458, PRE = 0.1595, REC = 0.5352
bus         : AP = 0.6788, PRE = 0.1664, REC = 0.9108
car         : AP = 0.7422, PRE = 0.2731, REC = 0.8426
cat         : AP = 0.7405, PRE = 0.3226, REC = 0.8715
chair       : AP = 0.3686, PRE = 0.1081, REC = 0.7222
cow         : AP = 0.6950, PRE = 0.2007, REC = 0.8975
diningtable : AP = 0.5186, PRE = 0.1034, REC = 0.8495
dog         : AP = 0.7578, PRE = 0.2607, REC = 0.9121
horse       : AP = 0.7637, PRE = 0.2305, REC = 0.9023
motorbike   : AP = 0.7038, PRE = 0.2657, REC = 0.8462
person      : AP = 0.7207, PRE = 0.3107, REC = 0.8357
pottedplant : AP = 0.3222, PRE = 0.0976, REC = 0.6458
sheep       : AP = 0.5389, PRE = 0.2365, REC = 0.7438
sofa        : AP = 0.5286, PRE = 0.1366, REC = 0.8996
train       : AP = 0.6922, PRE = 0.1896, REC = 0.8901
tvmonitor   : AP = 0.6377, PRE = 0.1815, REC = 0.8084
Mean AP = 0.6064
```
# Acknol