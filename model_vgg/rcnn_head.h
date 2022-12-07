#pragma once

#include "bbox.h"
#include "example.h"
#include "loss.h"

#include <string_view>
#include <vector>

#include <boost/json/value.hpp>
#include <torch/torch.h>

namespace rcnn_head
{
/**
  do RoIPool or RoIAlign on different levels.
 */
class RoIExtractorImpl : public torch::nn::Module
{
  public:
    enum class roi_type
    {
        RoIAlign,
        RoIPool
    };
    RoIExtractorImpl(int featmap_strides,          // size must be smaller than number of features recieved
                     std::string_view type,        // support either RoIAlign or RoIPool
                     std::vector<int> output_size, // [h, w], e.g. [7, 7]
                     int sampling_ratio = 0);      // only used in RoIAlign

    RoIExtractorImpl(const boost::json::value &opts);

    // return fix-sized roi pooling result
    torch::Tensor forward(torch::Tensor feat, torch::Tensor rois);

  private:
    int _featmap_strides;
    roi_type _roi_type{roi_type::RoIAlign};
    std::vector<int> _output_size;
    int _sampling_ratio;
};
TORCH_MODULE(RoIExtractor);

/**
   Typically RCNN consists of two fc layers followed by two parallel fc layers for classification and regression.
 */
class RCNNHeadImpl : public torch::nn::Module
{
  public:
    RCNNHeadImpl(int in_channels, const std::vector<long> &fc_out_channels, int num_classes, int roi_feat_size,
                 const boost::json::value &roi_extractor_opts, const boost::json::value &bbox_coder_opts,
                 const boost::json::value &loss_cls_opts, const boost::json::value &loss_bbox_opts,
                 const boost::json::value &train_opts, const boost::json::value &test_opts);

    RCNNHeadImpl(const boost::json::value &opts);

    void set_fcs(torch::nn::Sequential fcs);

    // return cls_outs and bbox_outs
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor feat, torch::Tensor rois);
    // return loss_cls and loss_bbox
    std::tuple<torch::Tensor, torch::Tensor> forward_train(torch::Tensor feat, torch::Tensor rois,
                                                           const dataset::DetectionExample &example);
    // return det results
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test(torch::Tensor feat, torch::Tensor rois,
                                                                         const dataset::DetectionExample &example);

  private:
    int _num_classes;

    const boost::json::value &_test_opts;

    bbox::BBoxRegressCoder _bbox_coder;
    bbox::BBoxAssigner _bbox_assigner;

    RoIExtractor _roi_extractor{nullptr};

    torch::nn::Sequential _shared_fcs;
    torch::nn::Linear _classifier{nullptr};
    torch::nn::Linear _regressor{nullptr};
    std::shared_ptr<loss::Loss> _loss_cls{nullptr};
    std::shared_ptr<loss::Loss> _loss_bbox{nullptr};
};
TORCH_MODULE(RCNNHead);

} // namespace rcnn_head
