#include "rpn_head.h"
#include "anchor.h"

#include <torchvision/ops/nms.h>

#include <array>
#include <cassert>
#include <tuple>

namespace rpn_head
{
using namespace torch::indexing;

RPNHeadImpl::RPNHeadImpl(const std::int64_t in_channel, const std::int64_t out_channel,
                         const boost::json::value &anchor_opts, const boost::json::value &bbox_coder_opts,
                         const boost::json::value &loss_cls_opts, const boost::json::value &loss_bbox_opts,
                         const boost::json::value &train_opts, const boost::json::value &test_opts)
    : _train_opts{train_opts}, _test_opts{test_opts}, _bbox_coder{bbox_coder_opts}, _anchor_generator{anchor_opts},
      _bbox_assigner{train_opts.at("bbox_assigner_opts")}
{
    _class_channels = 1; // use sigmoid, so class channel is 1, or 2 when softmax
    _conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, 3).stride(1).padding(1));
    _classifier = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channel, _anchor_generator->num_anchors() * _class_channels /* 9 */, 1));
    _regressor = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channel, _anchor_generator->num_anchors() * 4, 1));
    _loss_cls = loss::build_loss(loss_cls_opts);
    _loss_bbox = loss::build_loss(loss_bbox_opts);
    register_module("conv", _conv);
    register_module("classifier", _classifier);
    register_module("regressor", _regressor);
    register_module("anchor_generator", _anchor_generator);
    register_module("loss_cls", _loss_cls);
    register_module("loss_bbox", _loss_bbox);

    // init_weights
    torch::nn::init::normal_(_conv->weight, 0, 0.01);
    torch::nn::init::normal_(_classifier->weight, 0, 0.01);
    torch::nn::init::normal_(_regressor->weight, 0, 0.01);
    torch::nn::init::constant_(_conv->bias, 0);
    torch::nn::init::constant_(_classifier->bias, 0);
    torch::nn::init::constant_(_regressor->bias, 0);
}

RPNHeadImpl::RPNHeadImpl(const boost::json::value &opts)
    : RPNHeadImpl(opts.at("in_channels").as_int64(), opts.at("out_channels").as_int64(), opts.at("anchor_opts"),
                  opts.at("bbox_coder_opts"), opts.at("loss_cls_opts"), opts.at("loss_bbox_opts"),
                  opts.at("train_opts"), opts.at("test_opts"))
{
}

std::tuple<torch::Tensor, torch::Tensor> RPNHeadImpl::forward(torch::Tensor feat) // [n, 512, h / 16, w / 16]
{
    torch::Tensor cls_outs, bbox_outs; // cls/bbox outputs for the feature
    auto x = _conv->forward(feat).relu_();
    cls_outs = _classifier->forward(x);
    bbox_outs = _regressor->forward(x);
    return std::make_tuple(cls_outs, bbox_outs);
}

// return cls_loss, bbox_loss and proposals
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RPNHeadImpl::forward_train(
    torch::Tensor feat, const dataset::DetectionExample &example)
{
    auto batch_size = feat.size(0); // 1
    // feat has shape: [1, 512, feat_h = h / 16, feat_w = w / 16], e.g. [1, 512, 38, 58]
    auto rpn_outs = forward(feat);

    // cls_out has shape: [n, num_anchors*classs_channels, feat_h, feat_w], e.g. [1, 9, 38, 58]
    auto cls_out = std::get<0>(rpn_outs);

    // bbox_out has shape: [n, num_anchors*4, feat_h, feat_w], e.g. [1, 36, 38, 58]
    auto bbox_out = std::get<1>(rpn_outs);

    // anchors has shape:  [grid_y, grid_x, num_scales, num_ratios, 4], e.g. [38, 58, 3, 3, 4]
    auto anchors = _anchor_generator->get_anchors({feat.size(2), feat.size(3)} /* {feat_h, feat_w}, e.g. {38, 58}*/);

    // next unify the shapes
    cls_out = cls_out.permute({0, 2, 3, 1}); // [n, grid_h, grid_w, num_anchors * classs_channels], e.g. [1, 38, 58, 9]
    bbox_out = bbox_out.permute({0, 2, 3, 1}); // [n, grid_h, grid_w, num_anchors * 4], e.g. [1, 38, 58, 36]
    cls_out = cls_out.reshape(
        {-1, _class_channels});           // [n * grid_h * grid_w * num_anchors, classs_channels], e.g. [1 * 38 * 58, 1]
    bbox_out = bbox_out.reshape({-1, 4}); // [n * grid_h * grid_w * num_anchors, 4], e.g. [1 * 38 * 58, 4]
    anchors = anchors.reshape({-1, 4});   // [grid_h * grid_w * num_anchors, 4], e.g. [38 * 58 * 3 * 3, 4]

    /*************** get proposals *************/
    auto proposals = get_proposals(anchors, cls_out, bbox_out, example.img_shape, _train_opts);

    // get general assigned results
    auto assigned_result = _bbox_assigner.assign(anchors, example.target.gt_bboxes, example.target.gt_labels);
    auto &assigned_inds = assigned_result.assigned_inds;
    auto chosen = (assigned_inds >= -1);
    auto tar_inds = assigned_inds.index({chosen});
    auto pos_mask = (tar_inds >= 0);

    // next calc cls loss
    auto pred_cls = cls_out.index({chosen}).view(-1);
    auto tar_labels = pos_mask.to(torch::kInt64); // special case for RPN
    auto loss_cls = _loss_cls->forward(pred_cls, tar_labels.to(torch::kFloat32), tar_inds.size(0));

    // next calc reg loss
    auto tar_anchors = assigned_result.bboxes.index({chosen});
    // negative bboxes are assigned with the last gt(-1), they will be ignored anyway
    auto tar_bboxes = assigned_result.gt_bboxes.index({tar_inds});

    auto bbox_pred = torch::Tensor(), bbox_tar = torch::Tensor();
    if (std::dynamic_pointer_cast<loss::GIoULoss>(_loss_bbox) != nullptr)
    {
        bbox_tar = tar_bboxes;
        bbox_pred = _bbox_coder.decode(tar_anchors, bbox_out.index({chosen}));
    }
    else
    {
        bbox_tar = _bbox_coder.encode(tar_anchors, tar_bboxes);
        bbox_pred = bbox_out.index({chosen});
    }
    auto loss_bbox = _loss_bbox->forward(bbox_pred.index({pos_mask}), bbox_tar.index({pos_mask}), tar_inds.size(0));

    return std::make_tuple(loss_cls, loss_bbox, proposals);
}

/*
 *  method get_proposals()
 *
 */
// assume opts defines
//   nms_pre: number of proposals to select for each feature level
//   nms_post: number of proposals to select after batched nms
//   nms_thr: iou_thr to be used in NMS
//   min_bbox_size: filter bbox with size < min_bbox_size
torch::Tensor RPNHeadImpl::get_proposals(torch::Tensor anchors,    // [grid_h * grid_w * num_anchors, 4]
                                         torch::Tensor cls_out,    // [n * grid_h * grid_w * num_anchors, 1]
                                         torch::Tensor bbox_delta, // [n * grid_h * grid_w * num_anchors, 4]
                                         const std::vector<int64_t> &img_shape, // (h, w)
                                         const boost::json::value &opts)
{
    assert(cls_out.size(0) == bbox_delta.size(0) && "cls_outs and bbox_delta must have same number of levels");

    auto cls_score = cls_out.view(-1).sigmoid();
    if (auto nms_pre = opts.at("nms_pre").as_int64(); nms_pre < cls_score.size(0))
    {
        auto topk_inds = std::get<1>(cls_score.topk(nms_pre));
        cls_score = cls_score.index({topk_inds});
        bbox_delta = bbox_delta.index({topk_inds});
        anchors = anchors.index({topk_inds});
    }

    auto bbox = _bbox_coder.decode(anchors, bbox_delta, img_shape);

    if (auto min_bbox_size = opts.at("min_bbox_size").as_int64(); min_bbox_size > 0)
    {
        auto large_mask = ((bbox.index({Slice(), 2}) - bbox.index({Slice(), 0})) >= min_bbox_size) &
                          ((bbox.index({Slice(), 3}) - bbox.index({Slice(), 1})) >= min_bbox_size);
        cls_score = cls_score.index({large_mask});
        bbox = bbox.index({large_mask});
    }

    auto keep = vision::ops::nms(bbox, cls_score, opts.at("nms_thr").as_double());
    // proposals are [n, 5] tensors where last index is score
    auto proposals = torch::cat({bbox.index({keep}), cls_score.index({keep}).view({-1, 1})}, 1);
    if (auto nms_post = opts.at("nms_post").as_int64(); nms_post < proposals.size(0))
    {
        proposals = proposals.index({Slice(None, nms_post)});
    }
    return proposals;
}

torch::Tensor RPNHeadImpl::forward_test(torch::Tensor feat, const dataset::DetectionExample &example)
{
    auto rpn_outs = forward(feat);
    auto cls_out = std::get<0>(rpn_outs), bbox_out = std::get<1>(rpn_outs);
    auto anchors = _anchor_generator->get_anchors({feat.size(2), feat.size(3)});

    cls_out = cls_out.permute({0, 2, 3, 1});          // [n, grid_h, grid_w, num_anchors * classs_channels]
    bbox_out = bbox_out.permute({0, 2, 3, 1});        // [n, grid_h, grid_w, num_anchors * 4]
    cls_out = cls_out.reshape({-1, _class_channels}); // [n * grid_h * grid_w * num_anchors, classs_channels]
    bbox_out = bbox_out.reshape({-1, 4});             // [n * grid_h * grid_w * num_anchors, 4]
    anchors = anchors.reshape({-1, 4});               // [grid_h * grid_w * num_anchors, 4]

    /*************** get proposals *************/
    return get_proposals(anchors, cls_out, bbox_out, example.img_shape, _test_opts);
}

} // namespace rpn_head
