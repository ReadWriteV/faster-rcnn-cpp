#pragma once

#include "anchor.h"
#include "bbox.h"
#include "example.h"
#include "loss.h"

#include <boost/json/value.hpp>
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
    RPNHeadImpl(const std::int64_t in_channel, const std::int64_t out_channel, const boost::json::value &anchor_opts,
                const boost::json::value &bbox_coder_opts, const boost::json::value &loss_cls_opts,
                const boost::json::value &loss_bbox_opts, const boost::json::value &train_opts,
                const boost::json::value &test_opts);

    RPNHeadImpl(const boost::json::value &opts);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor feat);

    // return cls_loss, bbox_loss and proposals
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_train(torch::Tensor feat,
                                                                          const dataset::DetectionExample &example);

    // return proposals
    torch::Tensor forward_test(torch::Tensor feat, const dataset::DetectionExample &example);

    // get_proposal
    torch::Tensor get_proposals(torch::Tensor anchors, torch::Tensor cls_out, torch::Tensor bbox_delta,
                                const std::vector<int64_t> &img_shape, const boost::json::value &opts);

  private:
    const boost::json::value &_train_opts;
    const boost::json::value &_test_opts;

    bbox::BBoxRegressCoder _bbox_coder;
    anchor::AnchorGenerator _anchor_generator;
    bbox::BBoxAssigner _bbox_assigner;

    std::uint32_t _class_channels{1};

    torch::nn::Conv2d _conv{nullptr};
    torch::nn::Conv2d _classifier{nullptr};
    torch::nn::Conv2d _regressor{nullptr};
    std::shared_ptr<loss::Loss> _loss_cls{nullptr};
    std::shared_ptr<loss::Loss> _loss_bbox{nullptr};
};
TORCH_MODULE(RPNHead);

} // namespace rpn_head