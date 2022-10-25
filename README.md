# Faster RCNN Cpp

Faster RCNN、DA Faster RCNN的 libtorch 实现。

## Feature

+ VOC 和 COCO 两种常见数据集实现
+ VOC 评价方法实现
+ VGG 系列和 ResNet 系列特征提取网络的实现

## Todo

+ 数据集变换采用 libtorch 的 *Transform* 接口
+ boost::property_tree::ptree 实现 *get&lt;std::vector&lt;T>>*

## Benchmark

PASCAL VOC 2007 (Train/Test: 07trainval/07test, non-difficult, ROI Align)

### ResNet50_FPN (COCO Metric and VOC Metric)

| COCO Metric | AP | AP50 |
| ---------- | -------- | ---------- |
|  this  |  0.439   | 0.767 |
| mmdet | 0.437 | 0.769 |
|base |0.438 | 0.768|
|best|0.439 | 0.771|

|    VOC Metric        | AP |
| ---------- | -------- |
|  this  |  0.7545   |
|  best  |  0.7606   |

### VGG16 (VOC Metric)

|    this        | AP |
| ---------- | -------- |
| VGG-feature + VGG-classifier + old-RPN without flipped  |  0.6064   |
| VGG-feature + classifier + old-RPN without flipped  |  0.6405   |


|    Pytorch        | AP |
| ---------- | -------- |
|  without flipped  |  0.674   |
|  with flipped  |  0.7010   |






## Acknowledgement
This project is based on the following projects:

+ https://github.com/thisisi3/libtorch-faster-rcnn
+ https://github.com/jwyang/faster-rcnn.pytorch
