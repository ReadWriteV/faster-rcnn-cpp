#include "rcnn_head.h"
#include "utils.h"

#include <cassert>
#include <torchvision/ops/roi_align.h>
#include <torchvision/ops/roi_pool.h>
#include <vector>

namespace rcnn_head
{
using namespace torch::indexing;

RoIExtractorImpl::RoIExtractorImpl(int out_channels, const std::vector<int> &featmap_strides, const std::string &type,
                                   const std::vector<int> &output_size, int sampling_ratio, int finest_scale)
    : _out_channels(out_channels), _featmap_strides(featmap_strides), _type(type), _output_size(output_size),
      _sampling_ratio(sampling_ratio), _finest_scale(finest_scale)
{
    assert(type == "RoIAlign" || type == "RoIPool" && "RoIExtractor only support type RoIAlign or RoIPool");
    assert(output_size.size() == 2 && "RoIExtractor's output_size must be of size 2");
    if (type == "RoIPool")
    {
        _roi_type = roi_type::RoIPool;
    }
    assert(featmap_strides.size() > 0 && "number of featmap should be strictly positive");
}

RoIExtractorImpl::RoIExtractorImpl(const boost::property_tree::ptree &opts)
{
    _out_channels = opts.get<int>("out_channels");
    const auto &featmap_strides_node = opts.get_child("featmap_strides");
    std::transform(featmap_strides_node.begin(), featmap_strides_node.end(), std::back_inserter(_featmap_strides),
                   [](const boost::property_tree::ptree::value_type &v) { return v.second.get_value<int>(); });
    _type = opts.get<std::string>("type");

    const auto &output_size_node = opts.get_child("output_size");
    std::transform(output_size_node.begin(), output_size_node.end(), std::back_inserter(_output_size),
                   [](const boost::property_tree::ptree::value_type &v) { return v.second.get_value<int>(); });
    _sampling_ratio = opts.get<int>("sampling_ratio");
    _finest_scale = opts.get<int>("finest_scale");

    assert(_type == "RoIAlign" || _type == "RoIPool" && "RoIExtractor only support type RoIAlign or RoIPool");
    assert(_output_size.size() == 2 && "RoIExtractor's output_size must be of size 2");
    if (_type == "RoIPool")
    {
        _roi_type = roi_type::RoIPool;
    }
    assert(_featmap_strides.size() > 0 && "number of featmap should be strictly positive");
}

torch::Tensor RoIExtractorImpl::forward(std::vector<torch::Tensor> feats, torch::Tensor rois)
{
    assert(feats.size() >= _featmap_strides.size() && "number of feats must not be less than that of featmap_strides");
    if (rois.size(1) > 4)
    {
        rois = rois.index({Slice(), Slice(None, 4)});
    }
    int num_lvls = _featmap_strides.size();
    auto tsr_opts = feats[0].options();
    auto roi_feats = torch::full({rois.size(0), _out_channels, _output_size[0], _output_size[1]}, 0.0, tsr_opts);
    auto tar_lvls = map_roi_levels(rois, num_lvls);
    for (int i = 0; i < num_lvls; i++)
    {
        auto inds = (tar_lvls == i);
        if (inds.any().item<bool>())
        {
            torch::Tensor roi_feat;
            auto idx_rois = rois.index({inds});
            // add image idx required by torchvision' roi_align
            idx_rois = torch::cat({torch::full({idx_rois.size(0), 1}, 0, tsr_opts), idx_rois}, 1);
            if (_roi_type == roi_type::RoIAlign)
            {
                roi_feat = vision::ops::roi_align(feats[i], idx_rois, 1.0 / _featmap_strides[i], _output_size[0],
                                                  _output_size[1], _sampling_ratio, true);
            }
            else
            {
                auto pool_res = vision::ops::roi_pool(feats[i], idx_rois, 1.0 / _featmap_strides[i], _output_size[0],
                                                      _output_size[1]);
                roi_feat = std::get<0>(pool_res);
            }
            roi_feats.index_put_({inds}, roi_feat);
        }
    }
    return roi_feats;
}

torch::Tensor RoIExtractorImpl::map_roi_levels(torch::Tensor rois, int num_levels)
{
    auto scale = utils::bbox_area(rois).sqrt();
    auto tar_lvls = torch::log2(scale / _finest_scale + 1e-6).floor();
    return tar_lvls.clamp(0, num_levels - 1).to(torch::kLong);
}

/**
   class RCNNHead

 */
RCNNHeadImpl::RCNNHeadImpl(int in_channels, const std::vector<int> &fc_out_channels, int num_classes, int roi_feat_size,
                           const boost::property_tree::ptree &roi_extractor_opts,
                           const boost::property_tree::ptree &bbox_coder_opts,
                           const boost::property_tree::ptree &loss_cls_opts,
                           const boost::property_tree::ptree &loss_bbox_opts,
                           const boost::property_tree::ptree &train_opts, const boost::property_tree::ptree &test_opts)
    : _in_channels(in_channels), _fc_out_channels(fc_out_channels), _num_classes(num_classes),
      _roi_feat_size(roi_feat_size), _roi_extractor_opts(roi_extractor_opts), _bbox_coder_opts(bbox_coder_opts),
      _loss_cls_opts(loss_cls_opts), _loss_bbox_opts(loss_bbox_opts), _train_opts(train_opts), _test_opts(test_opts),
      _bbox_coder(bbox_coder_opts), _bbox_assigner(train_opts.get_child("bbox_assigner_opts")),
      _roi_extractor(roi_extractor_opts)
{
    init();
}

RCNNHeadImpl::RCNNHeadImpl(const boost::property_tree::ptree &opts)
    : _in_channels(opts.get<int>("in_channels")), _num_classes(opts.get<int>("num_classes")),
      _roi_feat_size(opts.get<int>("roi_feat_size")), _roi_extractor_opts(opts.get_child("roi_extractor_opts")),
      _bbox_coder_opts(opts.get_child("bbox_coder_opts")), _loss_cls_opts(opts.get_child("loss_cls_opts")),
      _loss_bbox_opts(opts.get_child("loss_bbox_opts")), _train_opts(opts.get_child("train_opts")),
      _test_opts(opts.get_child("test_opts")), _bbox_coder(_bbox_coder_opts),
      _bbox_assigner(_train_opts.get_child("bbox_assigner_opts")), _roi_extractor(_roi_extractor_opts)
{
    const auto &fc_out_channels_node = opts.get_child("fc_out_channels");
    std::transform(fc_out_channels_node.begin(), fc_out_channels_node.end(), std::back_inserter(_fc_out_channels),
                   [](const boost::property_tree::ptree::value_type &v) { return v.second.get_value<int>(); });

    init();
}

void RCNNHeadImpl::init()
{
    int fc_in = _in_channels * _roi_feat_size * _roi_feat_size;
    _shared_fcs = torch::nn::ModuleList();
    for (auto fc_out : _fc_out_channels)
    {
        _shared_fcs->push_back(torch::nn::Linear(fc_in, fc_out));
        fc_in = fc_out;
    }
    _classifier = torch::nn::Linear(_fc_out_channels.back(), _num_classes + 1);
    _regressor = torch::nn::Linear(_fc_out_channels.back(), _num_classes * 4);

    _loss_cls = loss::build_loss(_loss_cls_opts);
    _loss_bbox = loss::build_loss(_loss_bbox_opts);

    register_module("roi_extractor", _roi_extractor);
    register_module("shared_fc", _shared_fcs);
    register_module("classifier", _classifier);
    register_module("regressor", _regressor);
    register_module("loss_cls", _loss_cls);
    register_module("loss_bbox", _loss_bbox);

    // init weights
    for (int i = 0; i < _fc_out_channels.size(); i++)
    {
        torch::nn::init::xavier_uniform_(_shared_fcs[i]->as<torch::nn::Linear>()->weight);
        torch::nn::init::constant_(_shared_fcs[i]->as<torch::nn::Linear>()->bias, 0.0);
    }
    torch::nn::init::normal_(_classifier->weight, 0, 0.01);
    torch::nn::init::normal_(_regressor->weight, 0, 0.001);
    torch::nn::init::constant_(_classifier->bias, 0.0);
    torch::nn::init::constant_(_regressor->bias, 0.0);
}

std::tuple<torch::Tensor, torch::Tensor> RCNNHeadImpl::forward(std::vector<torch::Tensor> feats, torch::Tensor rois)
{
    auto roi_feats = _roi_extractor->forward(feats, rois);
    auto x = roi_feats.view({roi_feats.size(0), -1});
    for (int i = 0; i < _shared_fcs->size(); i++)
    {
        x = _shared_fcs[i]->as<torch::nn::Linear>()->forward(x).relu_();
    }
    auto cls_outs = _classifier->forward(x);
    auto bbox_outs = _regressor->forward(x);
    return std::make_tuple(cls_outs, bbox_outs);
}

std::tuple<torch::Tensor, torch::Tensor> RCNNHeadImpl::forward_train(std::vector<torch::Tensor> feats,
                                                                     torch::Tensor rois, // [n, 5]
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
    auto rcnn_outs = forward(feats, tar_rois);
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
    if (_loss_bbox_opts.get<std::string>("type") == "GIoULoss")
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
    std::vector<torch::Tensor> feats, torch::Tensor rois, const dataset::DetectionExample &example)
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
    auto rcnn_outs = forward(feats, rois);
    auto cls_outs = std::get<0>(rcnn_outs),
         bbox_outs = std::get<1>(rcnn_outs);                                 // cls_outs: [n, 21], bbox_outs: [n, 20*4]
    auto cls_scores = cls_outs.softmax(1).index({Slice(), Slice(None, -1)}); // filter out bg label which=num_classes
    rois = rois.view({-1, 1, 4}).repeat({1, _num_classes, 1}).view({-1, 4});

    auto pred_bboxes = _bbox_coder.decode(rois, bbox_outs.view({-1, 4}), example.img_shape);
    pred_bboxes = pred_bboxes.view({num_rois, -1});
    auto nms_res = bbox::multiclass_nms(pred_bboxes, cls_scores, 0.5, 0.05, 100);
    return nms_res; // bboxes, scores, labels
}

} // namespace rcnn_head
