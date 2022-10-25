#include "detector.h"
#include "vgg.h"
#include "voc.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>

namespace detector
{
FasterRCNNVGG16Impl::FasterRCNNVGG16Impl(const boost::property_tree::ptree &backbone_opts,
                                         const boost::property_tree::ptree &rpn_opts,
                                         const boost::property_tree::ptree &rcnn_opts)
{
    assert(backbone_opts.get<std::string>("type") == "vgg16");
    auto vgg16 = backbone::make_vgg16(1000, true);

    rpn = rpn_head::RPNHead(rpn_opts);
    rcnn = rcnn_head::RCNNHead(rcnn_opts);

    // weight initialization is included in constructor for all modules except for
    // initializing backbone with ImageNet pretrained weight
    std::string pretrained = backbone_opts.get<std::string>("pretrained");
    std::cout << "loading weights for backbone...\n";
    torch::load(vgg16, pretrained);

    auto features_sequential = vgg16->features->modules(false);
    // remove last max pooling layer
    features_sequential.pop_back();

    /// I have no idea why the following code cannot be compiled, so I use dynamic cast
    /// ```
    /// for (auto &module : features_sequential)
    /// {
    ///     feature_extractor->push_back(M);
    /// }
    /// ```
    for (auto &module : features_sequential)
    {
        if (auto M = std::dynamic_pointer_cast<torch::nn::MaxPool2dImpl>(module))
        {
            feature_extractor->push_back(M);
        }
        else if (auto M = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(module))
        {
            feature_extractor->push_back(M);
        }
        else if (auto M = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(module))
        {
            feature_extractor->push_back(M);
        }
        else
        {
            throw std::runtime_error("unknown layer");
        }
    }

    // Fix the layers before conv3:
    for (int i = 0; i < 10; i++)
    {
        auto ptr = feature_extractor[i].get();
        ptr->eval();
        for (auto &para : ptr->parameters())
        {
            para.requires_grad_(false);
        }
    }

    // this should be in RCNNHead
    // torch::nn::Sequential RCNN_top;
    // for (std::size_t i = 0; i < vgg16->classifier->size() - 1; i++)
    // {
    //     RCNN_top->push_back(vgg16->classifier[i]);
    // }
    // rcnn->set_fcs(vgg16->classifier);

    register_module("feature_extractor", feature_extractor);
    register_module("rpn_head", rpn);
    register_module("rcnn_head", rcnn);
}

// return a map/dict of losses
std::map<std::string, torch::Tensor> FasterRCNNVGG16Impl::forward_train(const dataset::DetectionExample &example)
{
    auto feat = feature_extractor->forward(example.data);
    auto rpn_outs = rpn->forward_train(feat, example);
    auto rpn_cls_loss = std::get<0>(rpn_outs), rpn_bbox_loss = std::get<1>(rpn_outs), proposals = std::get<2>(rpn_outs);
    auto rcnn_outs = rcnn->forward_train(feat, proposals, example);
    auto rcnn_cls_loss = std::get<0>(rcnn_outs), rcnn_bbox_loss = std::get<1>(rcnn_outs);
    return {{"rpn_cls_loss", rpn_cls_loss},
            {"rpn_bbox_loss", rpn_bbox_loss},
            {"rcnn_cls_loss", rcnn_cls_loss},
            {"rcnn_bbox_loss", rcnn_bbox_loss}};
}

// return det_bboxes, det_scores, det_labels
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FasterRCNNVGG16Impl::forward_test(
    const dataset::DetectionExample &img_data)
{
    auto feat = feature_extractor->forward(img_data.data);
    auto proposals = rpn->forward_test(feat, img_data);
    auto det_res = rcnn->forward_test(feat, proposals, img_data);
    return det_res;
}

} // namespace detector