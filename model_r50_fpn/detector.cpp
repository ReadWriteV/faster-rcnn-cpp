#include "detector.h"
#include <string>

namespace detector
{
FasterRCNNImpl::FasterRCNNImpl(const boost::property_tree::ptree &backbone_opts,
                               const boost::property_tree::ptree &fpn_opts, const boost::property_tree::ptree &rpn_opts,
                               const boost::property_tree::ptree &rcnn_opts)
    : _backbone_opts(backbone_opts), _fpn_opts(fpn_opts), _rpn_opts(rpn_opts), _rcnn_opts(rcnn_opts)
{
    // building backbones is different as it allows different types of resnet
    _backbone = backbone::build_backbone(_backbone_opts);
    _neck = neck::FPN(_fpn_opts);
    _rpn_head = rpn_head::RPNHead(_rpn_opts);
    _rcnn_head = rcnn_head::RCNNHead(_rcnn_opts);

    // weight initialization is included in constructor for all modules except for
    // initializing backbone with ImageNet pretrained weight
    std::string pretrained = backbone_opts.get<std::string>("pretrained");
    std::cout << "loading weights for backbone...\n";
    torch::load(_backbone, pretrained);

    register_module("backbone", _backbone);
    register_module("neck", _neck);
    register_module("rpn_head", _rpn_head);
    register_module("rcnn_head", _rcnn_head);
}

// return a map/dict of losses
std::map<std::string, torch::Tensor> FasterRCNNImpl::forward_train(const dataset::DetectionExample &example)
{
    auto feats = _backbone->forward(example.data);
    feats = _neck->forward(feats);
    auto rpn_outs = _rpn_head->forward_train(feats, example);
    auto rpn_cls_loss = std::get<0>(rpn_outs), rpn_bbox_loss = std::get<1>(rpn_outs), proposals = std::get<2>(rpn_outs);
    auto rcnn_outs = _rcnn_head->forward_train(feats, proposals, example);
    auto rcnn_cls_loss = std::get<0>(rcnn_outs), rcnn_bbox_loss = std::get<1>(rcnn_outs);
    return {{"rpn_cls_loss", rpn_cls_loss},
            {"rpn_bbox_loss", rpn_bbox_loss},
            {"rcnn_cls_loss", rcnn_cls_loss},
            {"rcnn_bbox_loss", rcnn_bbox_loss}};
}

// return det_bboxes, det_scores, det_labels
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FasterRCNNImpl::forward_test(
    const dataset::DetectionExample &img_data)
{
    auto feats = _backbone->forward(img_data.data);
    feats = _neck->forward(feats);
    auto proposals = _rpn_head->forward_test(feats, img_data);
    auto det_res = _rcnn_head->forward_test(feats, proposals, img_data);
    return det_res;
}

} // namespace detector
