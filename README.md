# Faster RCNN Cpp

Faster RCNN 的 libtorch 实现。

## Feature

+ VOC 和 COCO 两种常见数据集实现
+ VOC 评价方法实现

## Todo

+ 数据集变换采用 libtorch 的 *Transform* 接口
+ 使用 boost::json 而非 boost::property_tree::ptree 加载配置文件

## Benchmark

PASCAL VOC 2007 (Train/Test: 07trainval/07test, non-difficult, ROI Align)

### ResNet50_FPN (COCO Metric and VOC Metric)

| COCO Metric | AP | AP50 |
| ---------- | -------- | ---------- |
|  this  |  0.439   | 0.767 |
| mmdet | 0.437 | 0.769 |
|  base  |0.438 | 0.768|

|    VOC Metric        | AP |
| ---------- | -------- |
|  this  |  0.7581   |
|  best  |  0.7606   |

### VGG16 (VOC Metric)

batch size: 1, lr: 0.00125, decay epoch: 9 12, total epoch: 12 

|    this        | AP |
| ---------- | -------- |
| VGG-feature + VGG-classifier + random-flipped |  0.6615   |
| VGG-feature + classifier + random-flipped |  0.6751   |
| VGG-feature + classifier + append-flipped |  0.6796   |

batch size: 1, lr: 1e-3, decay epoch: 5, total epoch: 6 

|    Pytorch        | AP |
| ---------- | -------- |
| VGG-feature + VGG-classifier + without flipped  |  0.674   |
| VGG-feature + VGG-classifier + without flipped + my-dataset  |  0.676   |
| VGG-feature + VGG-classifier + with flipped  |  0.7010   |






## Acknowledgement

This project is based on the following project:

+ https://github.com/thisisi3/libtorch-faster-rcnn
