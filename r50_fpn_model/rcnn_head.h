#pragma once

#include "bbox.h"
#include "example.h"
#include "loss.h"

#include <boost/property_tree/ptree.hpp>
#include <torch/torch.h>

namespace rcnn_head
{
/**
   Due to the use of FPN, the features are in multi-levels.
   It first maps rois to different feature levels by roi sizes,
   then do RoIPool or RoIAlign on different levels.
 */
class RoIExtractorImpl : public torch::nn::Module
{
  public:
    enum class roi_type
    {
        RoIAlign,
        RoIPool
    };
    RoIExtractorImpl(int out_channels,
                     const std::vector<int> &featmap_strides, // size must be smaller than number of features recieved
                     const std::string &type,                 // support either RoIAlign or RoIPool
                     const std::vector<int> &output_size,     // [h, w], e.g. [7, 7]
                     int sampling_ratio = 0,                  // only used in RoIAlign
                     int finest_scale = 56);

    RoIExtractorImpl(const boost::property_tree::ptree &opts);

    // return fix-sized roi pooling result
    torch::Tensor forward(std::vector<torch::Tensor> feats, torch::Tensor rois);
    // map rois to different feature levels by sizes of rois.
    torch::Tensor map_roi_levels(torch::Tensor rois, int num_levels);

  private:
    int _out_channels;
    std::vector<int> _featmap_strides;
    std::string _type;
    std::vector<int> _output_size;
    int _sampling_ratio;
    int _finest_scale;
    roi_type _roi_type{roi_type::RoIAlign};
};
TORCH_MODULE(RoIExtractor);

/**
   Typically RCNN consists of two fc layers followed by two parallel fc layers for classification and regression.
 */
class RCNNHeadImpl : public torch::nn::Module
{
  public:
    RCNNHeadImpl(int in_channels, const std::vector<int> &fc_out_channels, int num_classes, int roi_feat_size,
                 const boost::property_tree::ptree &roi_extractor_opts,
                 const boost::property_tree::ptree &bbox_coder_opts, const boost::property_tree::ptree &loss_cls_opts,
                 const boost::property_tree::ptree &loss_bbox_opts, const boost::property_tree::ptree &train_opts,
                 const boost::property_tree::ptree &test_opts);

    RCNNHeadImpl(const boost::property_tree::ptree &opts);

    void init();

    // return cls_outs and bbox_outs
    std::tuple<torch::Tensor, torch::Tensor> forward(std::vector<torch::Tensor> feats, torch::Tensor rois);
    // return loss_cls and loss_bbox
    std::tuple<torch::Tensor, torch::Tensor> forward_train(std::vector<torch::Tensor> feats, torch::Tensor rois,
                                                           const dataset::DetectionExample &example);
    // return det results
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test(std::vector<torch::Tensor> feats,
                                                                         torch::Tensor rois,
                                                                         const dataset::DetectionExample &example);

  private:
    int _in_channels;
    std::vector<int> _fc_out_channels;
    int _num_classes;
    int _roi_feat_size;
    boost::property_tree::ptree _roi_extractor_opts;
    boost::property_tree::ptree _bbox_coder_opts;
    boost::property_tree::ptree _train_opts;
    boost::property_tree::ptree _test_opts;
    boost::property_tree::ptree _loss_cls_opts;
    boost::property_tree::ptree _loss_bbox_opts;

    bbox::BBoxRegressCoder _bbox_coder;
    bbox::BBoxAssigner _bbox_assigner;

    RoIExtractor _roi_extractor{nullptr};
    torch::nn::ModuleList _shared_fcs{nullptr};
    torch::nn::Linear _classifier{nullptr};
    torch::nn::Linear _regressor{nullptr};
    std::shared_ptr<loss::Loss> _loss_cls{nullptr};
    std::shared_ptr<loss::Loss> _loss_bbox{nullptr};
};
TORCH_MODULE(RCNNHead);

} // namespace rcnn_head
