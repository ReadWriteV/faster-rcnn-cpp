#pragma once

#include "backbone.h"
#include "example.h"
#include "neck.h"
#include "rcnn_head.h"
#include "rpn_head.h"

#include <boost/property_tree/ptree.hpp>

namespace detector
{
/**
   Implementation of FasterRCNN
 */
class FasterRCNNImpl : public torch::nn::Module
{
  public:
    FasterRCNNImpl(const boost::property_tree::ptree &backbone_opts, const boost::property_tree::ptree &fpn_opts,
                   const boost::property_tree::ptree &rpn_opts, const boost::property_tree::ptree &rcnn_opts);
    // return a map of losses
    std::map<std::string, torch::Tensor> forward_train(const dataset::DetectionExample &img_data);
    // return det_bboxes, det_scores, det_labels
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test(const dataset::DetectionExample &img_data);

  private:
    const boost::property_tree::ptree &_backbone_opts;
    const boost::property_tree::ptree &_fpn_opts;
    const boost::property_tree::ptree &_rpn_opts;
    const boost::property_tree::ptree &_rcnn_opts;

    std::shared_ptr<backbone::Backbone> _backbone{nullptr};
    neck::FPN _neck{nullptr};
    rpn_head::RPNHead _rpn_head{nullptr};
    rcnn_head::RCNNHead _rcnn_head{nullptr};
};
TORCH_MODULE(FasterRCNN);

} // namespace detector
