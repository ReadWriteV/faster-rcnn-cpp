#pragma once

#include "anchor.h"
#include "bbox.h"
#include "example.h"
#include "loss.h"

#include <boost/property_tree/ptree.hpp>
#include <torch/torch.h>
#include <vector>

namespace rpn_head
{
/*
  Implementation of RPN.
 */
class RPNHeadImpl : public torch::nn::Module
{
  public:
    RPNHeadImpl(int in_channels, const boost::property_tree::ptree &anchor_opts,
                const boost::property_tree::ptree &bbox_coder_opts, const boost::property_tree::ptree &loss_cls_opts,
                const boost::property_tree::ptree &loss_bbox_opts, const boost::property_tree::ptree &train_opts,
                const boost::property_tree::ptree &test_opts);

    RPNHeadImpl(const boost::property_tree::ptree &opts);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor feat);

    // return cls_loss, bbox_loss and proposals
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_train(torch::Tensor feat,
                                                                          const dataset::DetectionExample &example);

    // return proposals
    torch::Tensor forward_test(torch::Tensor feat, const dataset::DetectionExample &example);

    // get_proposal
    torch::Tensor get_proposals(torch::Tensor anchors, torch::Tensor cls_out, torch::Tensor bbox_out,
                                const std::vector<int64_t> &img_shape, const boost::property_tree::ptree &opts);

  private:
    int _in_channels;
    boost::property_tree::ptree _anchor_opts;
    boost::property_tree::ptree _bbox_coder_opts;
    boost::property_tree::ptree _train_opts;
    boost::property_tree::ptree _test_opts;
    boost::property_tree::ptree _loss_cls_opts;
    boost::property_tree::ptree _loss_bbox_opts;

    bbox::BBoxRegressCoder _bbox_coder;
    anchor::AnchorGenerator _anchor_generator;
    bbox::BBoxAssigner _bbox_assigner;

    int _class_channels{1};

    torch::nn::Conv2d _conv{nullptr};
    torch::nn::Conv2d _classifier{nullptr};
    torch::nn::Conv2d _regressor{nullptr};
    std::shared_ptr<loss::Loss> _loss_cls{nullptr};
    std::shared_ptr<loss::Loss> _loss_bbox{nullptr};
};
TORCH_MODULE(RPNHead);

} // namespace rpn_head