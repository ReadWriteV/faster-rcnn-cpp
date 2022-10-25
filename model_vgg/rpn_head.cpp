#include "rpn_head.h"
#include "anchor.h"

#include <array>
#include <cassert>
#include <tuple>

namespace rpn_head
{
using namespace torch::indexing;

RPNHeadImpl::RPNHeadImpl(int in_channels, const boost::property_tree::ptree &anchor_opts,
                         const boost::property_tree::ptree &bbox_coder_opts,
                         const boost::property_tree::ptree &loss_cls_opts,
                         const boost::property_tree::ptree &loss_bbox_opts,
                         const boost::property_tree::ptree &train_opts, const boost::property_tree::ptree &test_opts)
    : _in_channels(in_channels), _anchor_opts(anchor_opts), _bbox_coder_opts(bbox_coder_opts),
      _loss_cls_opts(loss_cls_opts), _loss_bbox_opts(loss_bbox_opts), _train_opts(train_opts), _test_opts(test_opts),
      _bbox_coder(bbox_coder_opts), _anchor_generator(anchor_opts),
      _bbox_assigner(train_opts.get_child("bbox_assigner_opts"))
{
    _class_channels = 1; // use sigmoid, so class channel is 1, or 2 when softmax
    _conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(_in_channels, 512, 3).stride(1).padding(1));
    _classifier =
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, _anchor_generator->num_anchors() * _class_channels /* 9 */, 1));
    _regressor = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, _anchor_generator->num_anchors() * 4, 1));
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

RPNHeadImpl::RPNHeadImpl(const boost::property_tree::ptree &opts)
    : RPNHeadImpl(opts.get<int>("in_channels"), opts.get_child("anchor_opts"), opts.get_child("bbox_coder_opts"),
                  opts.get_child("loss_cls_opts"), opts.get_child("loss_bbox_opts"), opts.get_child("train_opts"),
                  opts.get_child("test_opts"))
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
    if (_loss_bbox_opts.get<std::string>("type") == "GIoULoss")
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
torch::Tensor RPNHeadImpl::get_proposals(torch::Tensor anchors,  // [grid_h * grid_w * num_anchors, 4]
                                         torch::Tensor cls_out,  // [n * grid_h * grid_w * num_anchors, 1]
                                         torch::Tensor bbox_out, // [n * grid_h * grid_w * num_anchors, 4]
                                         const std::vector<int64_t> &img_shape, // (h, w)
                                         const boost::property_tree::ptree &opts)
{
    assert(cls_out.size(0) == bbox_out.size(0) && "cls_outs and bbox_outs must have same number of levels");

    torch::Tensor score, delta, anchor, idx;

    auto cls_score = cls_out.view(-1).sigmoid();
    if (int nms_pre = opts.get<int>("nms_pre"); nms_pre < cls_score.size(0))
    {
        auto topk_inds = std::get<1>(cls_score.topk(nms_pre));
        cls_score = cls_score.index({topk_inds});
        bbox_out = bbox_out.index({topk_inds});
        anchors = anchors.index({topk_inds});
    }

    idx = torch::full({cls_score.size(0)}, 0, torch::kInt64).to(cls_score.device());

    auto all_bbox = _bbox_coder.decode(anchors, bbox_out, img_shape);

    if (int min_bbox_size = opts.get<int>("min_bbox_size"); min_bbox_size > 0)
    {
        auto large_mask = ((all_bbox.index({Slice(), 2}) - all_bbox.index({Slice(), 0})) >= min_bbox_size) &
                          ((all_bbox.index({Slice(), 3}) - all_bbox.index({Slice(), 1})) >= min_bbox_size);
        cls_score = cls_score.index({large_mask});
        all_bbox = all_bbox.index({large_mask});
        idx = idx.index({large_mask});
    }

    auto keep = bbox::batched_nms(all_bbox, cls_score, idx, opts.get<float>("nms_thr"));
    // proposals are [n, 5] tensors where last index is score
    auto proposals = torch::cat({all_bbox.index({keep}), cls_score.index({keep}).view({-1, 1})}, 1);
    if (int nms_post = opts.get<int>("nms_post"); nms_post < proposals.size(0))
    {
        proposals = proposals.index({Slice(None, nms_post)});
    }
    return proposals;
}

torch::Tensor RPNHeadImpl::forward_test(torch::Tensor feat, const dataset::DetectionExample &img_data)
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
    return get_proposals(anchors, cls_out, bbox_out, img_data.img_shape, _test_opts);
}

} // namespace rpn_head
