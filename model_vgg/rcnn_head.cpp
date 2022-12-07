#include "rcnn_head.h"
#include "utils.h"

#include <torchvision/ops/roi_align.h>
#include <torchvision/ops/roi_pool.h>

#include <boost/json.hpp>
#include <boost/json/serialize.hpp>
#include <cassert>
#include <vector>

namespace rcnn_head
{
using namespace torch::indexing;

RoIExtractorImpl::RoIExtractorImpl(int featmap_strides, std::string_view type, std::vector<int> output_size,
                                   int sampling_ratio)
    : _featmap_strides{featmap_strides}, _output_size{std::move(output_size)}, _sampling_ratio{sampling_ratio}
{
    assert(type == "RoIAlign" || type == "RoIPool" && "RoIExtractor only support type RoIAlign or RoIPool");
    assert(_output_size.size() == 2 && "RoIExtractor's output_size must be of size 2");
    if (type == "RoIPool")
    {
        _roi_type = roi_type::RoIPool;
    }
}

RoIExtractorImpl::RoIExtractorImpl(const boost::json::value &opts)
    : RoIExtractorImpl(opts.at("featmap_strides").as_int64(), opts.at("type").as_string().c_str(),
                       boost::json::value_to<std::vector<int>>(opts.at("output_size")),
                       opts.at("sampling_ratio").as_int64())
{
}

torch::Tensor RoIExtractorImpl::forward(torch::Tensor feat, torch::Tensor rois)
{
    if (rois.size(1) > 4)
    {
        rois = rois.index({Slice(), Slice(None, 4)});
    }
    auto tsr_opts = feat.options();
    torch::Tensor roi_feat;
    // add image idx required by torchvision' roi_align
    rois = torch::cat({torch::full({rois.size(0), 1}, 0, tsr_opts), rois}, 1);
    if (_roi_type == roi_type::RoIAlign)
    {
        roi_feat = vision::ops::roi_align(feat, rois, 1.0 / _featmap_strides, _output_size[0], _output_size[1],
                                          _sampling_ratio, true);
    }
    else
    {
        auto pool_res = vision::ops::roi_pool(feat, rois, 1.0 / _featmap_strides, _output_size[0], _output_size[1]);
        roi_feat = std::get<0>(pool_res);
    }
    return roi_feat;
}

/**
   class RCNNHead

 */
RCNNHeadImpl::RCNNHeadImpl(int in_channels, const std::vector<long> &fc_out_channels, int num_classes,
                           int roi_feat_size, const boost::json::value &roi_extractor_opts,
                           const boost::json::value &bbox_coder_opts, const boost::json::value &loss_cls_opts,
                           const boost::json::value &loss_bbox_opts, const boost::json::value &train_opts,
                           const boost::json::value &test_opts)
    : _num_classes(num_classes), _test_opts{test_opts}, _bbox_coder(bbox_coder_opts),
      _bbox_assigner(train_opts.at("bbox_assigner_opts")), _roi_extractor(roi_extractor_opts)
{
    std::int64_t fc_in = in_channels * roi_feat_size * roi_feat_size;
    for (auto fc_out : fc_out_channels)
    {
        auto linear = torch::nn::Linear(fc_in, fc_out);
        torch::nn::init::xavier_uniform_(linear->weight);
        torch::nn::init::constant_(linear->bias, 0.0);
        _shared_fcs->push_back(linear);
        _shared_fcs->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
        fc_in = fc_out;
    }
    _classifier = torch::nn::Linear(fc_out_channels.back(), _num_classes + 1);
    _regressor = torch::nn::Linear(fc_out_channels.back(), _num_classes * 4);

    _loss_cls = loss::build_loss(loss_cls_opts);
    _loss_bbox = loss::build_loss(loss_bbox_opts);

    register_module("roi_extractor", _roi_extractor);
    register_module("shared_fc", _shared_fcs);
    register_module("classifier", _classifier);
    register_module("regressor", _regressor);
    register_module("loss_cls", _loss_cls);
    register_module("loss_bbox", _loss_bbox);

    torch::nn::init::normal_(_classifier->weight, 0, 0.01);
    torch::nn::init::normal_(_regressor->weight, 0, 0.001);
    torch::nn::init::constant_(_classifier->bias, 0.0);
    torch::nn::init::constant_(_regressor->bias, 0.0);
}

RCNNHeadImpl::RCNNHeadImpl(const boost::json::value &opts)
    : RCNNHeadImpl(opts.at("in_channels").as_int64(),
                   boost::json::value_to<std::vector<long>>(opts.at("fc_out_channels")),
                   opts.at("num_classes").as_int64(), opts.at("roi_feat_size").as_int64(),
                   opts.at("roi_extractor_opts"), opts.at("bbox_coder_opts"), opts.at("loss_cls_opts"),
                   opts.at("loss_bbox_opts"), opts.at("train_opts"), opts.at("test_opts"))
{
}

void RCNNHeadImpl::set_fcs(torch::nn::Sequential fcs)
{
    this->_shared_fcs = replace_module("shared_fc", fcs);
}

std::tuple<torch::Tensor, torch::Tensor> RCNNHeadImpl::forward(torch::Tensor feat, torch::Tensor rois)
{
    auto roi_feats = _roi_extractor->forward(feat, rois);
    auto x = roi_feats.view({roi_feats.size(0), -1});
    x = _shared_fcs->forward(x);
    auto cls_outs = _classifier->forward(x);
    auto bbox_outs = _regressor->forward(x);
    return std::make_tuple(cls_outs, bbox_outs);
}

std::tuple<torch::Tensor, torch::Tensor> RCNNHeadImpl::forward_train(
    torch::Tensor feat, // feat has shape: [1, 512, feat_h = h / 16, feat_w = w / 16], e.g. [1, 512, 38, 58]
    torch::Tensor rois, // rois has shape: [n, 5]
    const dataset::DetectionExample &example)
{
    if (rois.size(1) > 4)
    {
        rois = rois.index({Slice(), Slice(None, 4)});
    }

    // get general assigned results
    auto assigned_result = _bbox_assigner.assign(rois, example.target.gt_bboxes, example.target.gt_labels);
    auto assigned_inds = assigned_result.assigned_inds; // -1 is bg, -2 is ignore
    auto chosen = (assigned_inds >= -1);
    auto tar_inds = assigned_inds.index({chosen});
    auto num_tars = tar_inds.size(0);
    auto pos_mask = (tar_inds >= 0);

    // apply forward to chosen rois
    auto tar_rois = assigned_result.bboxes.index({chosen}); // target rois, may contain GT
    auto rcnn_outs = forward(feat, tar_rois);
    auto cls_outs = std::get<0>(rcnn_outs),
         bbox_outs = std::get<1>(rcnn_outs); // cls_outs: [n, 21], bbox_outs: [n, 20*4]

    // next calc cls loss
    auto tar_labels = assigned_result.gt_labels.index({tar_inds});
    tar_labels.index_put_({torch::logical_not(pos_mask)}, _num_classes); // assign negative_labels=num_classes
    auto loss_cls = _loss_cls->forward(cls_outs, tar_labels.detach().clone(), num_tars);

    // next calc reg loss
    auto tar_bboxes = assigned_result.gt_bboxes.index({tar_inds}); // GTs to regress to
    tar_labels.index_put_({tar_labels == _num_classes}, -1);
    bbox_outs = bbox_outs.view({-1, _num_classes, 4});
    bbox_outs = bbox_outs.index({torch::arange(num_tars), tar_labels, Slice()}); // [n, 4]

    auto bbox_pred = torch::Tensor(), bbox_tar = torch::Tensor();
    if (std::dynamic_pointer_cast<loss::GIoULoss>(_loss_bbox) != nullptr)
    {
        bbox_tar = tar_bboxes;
        bbox_pred = _bbox_coder.decode(tar_rois, bbox_outs);
    }
    else
    {
        bbox_tar = _bbox_coder.encode(tar_rois, tar_bboxes);
        bbox_pred = bbox_outs;
    }
    auto loss_bbox =
        _loss_bbox->forward(bbox_pred.index({pos_mask}), bbox_tar.index({pos_mask}).detach().clone(), num_tars);
    return std::make_tuple(loss_cls, loss_bbox);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RCNNHeadImpl::forward_test(
    torch::Tensor feat, torch::Tensor rois, const dataset::DetectionExample &example)
{
    auto num_rois = rois.size(0);
    if (num_rois == 0)
    {
        return std::make_tuple(torch::empty({0, 4}), torch::empty({0}), torch::empty({0}));
    }
    if (rois.size(1) > 4)
    {
        rois = rois.index({Slice(), Slice(None, 4)});
    }
    auto rcnn_outs = forward(feat, rois);
    auto cls_outs = std::get<0>(rcnn_outs),
         bbox_outs = std::get<1>(rcnn_outs);                                 // cls_outs: [n, 21], bbox_outs: [n, 20*4]
    auto cls_scores = cls_outs.softmax(1).index({Slice(), Slice(None, -1)}); // filter out bg label which=num_classes
    rois = rois.view({-1, 1, 4}).repeat({1, _num_classes, 1}).view({-1, 4});

    auto pred_bboxes = _bbox_coder.decode(rois, bbox_outs.view({-1, 4}), example.img_shape);
    pred_bboxes = pred_bboxes.view({num_rois, -1});
    auto nms_res =
        bbox::multiclass_nms(pred_bboxes, cls_scores, _test_opts.at("nms_thr").as_double(),
                             _test_opts.at("score_thr").as_double(), _test_opts.at("max_per_img").as_int64());
    return nms_res; // bboxes, scores, labels
}

} // namespace rcnn_head
