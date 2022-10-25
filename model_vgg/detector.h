#pragma once

#include "example.h"
#include "rcnn_head.h"
#include "rpn_head.h"
#include "vgg.h"

#include <boost/property_tree/ptree.hpp>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>

namespace detector
{

/**
   Implementation of FasterRCNN with VGG16 as backbone
 */
class FasterRCNNVGG16Impl : public torch::nn::Module
{
  public:
    FasterRCNNVGG16Impl(const boost::property_tree::ptree &backbone_opts, const boost::property_tree::ptree &rpn_opts,
                        const boost::property_tree::ptree &rcnn_opts);
    // return a map of losses
    std::map<std::string, torch::Tensor> forward_train(const dataset::DetectionExample &img_data);
    // return det_bboxes, det_scores, det_labels
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test(const dataset::DetectionExample &img_data);

  private:
    torch::nn::Sequential feature_extractor;
    rpn_head::RPNHead rpn{nullptr};
    rcnn_head::RCNNHead rcnn{nullptr};
};
TORCH_MODULE(FasterRCNNVGG16);

} // namespace detector
