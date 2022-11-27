#include "anchor.h"
#include <cassert>

namespace anchor
{
/*
  class AnchorGenerator
 */

AnchorGeneratorImpl::AnchorGeneratorImpl(const std::vector<float> &anchor_scales,
                                         const std::vector<float> &anchor_ratios, const float &feat_stride)
    : _anchor_scales(anchor_scales), _anchor_ratios(anchor_ratios), _feat_stride(feat_stride)
{
    init();
}

AnchorGeneratorImpl::AnchorGeneratorImpl(const boost::property_tree::ptree &opts)
{
    auto anchor_scales_node = opts.get_child("anchor_scales");
    std::transform(anchor_scales_node.begin(), anchor_scales_node.end(), std::back_inserter(_anchor_scales),
                   [](const boost::property_tree::ptree::value_type &v) { return v.second.get_value<float>(); });

    auto anchor_ratios_node = opts.get_child("anchor_ratios");
    std::transform(anchor_ratios_node.begin(), anchor_ratios_node.end(), std::back_inserter(_anchor_ratios),
                   [](const boost::property_tree::ptree::value_type &v) { return v.second.get_value<float>(); });

    _feat_stride = opts.get<float>("feat_stride");

    init();
}

void AnchorGeneratorImpl::init()
{
    std::cout << "[AnchorGeneratorImpl::init] "
              << "anchor_scales: " << _anchor_scales << ", anchor_ratios" << _anchor_ratios
              << ", feat_stride: " << _feat_stride << std::endl;

    auto scales = torch::tensor(_anchor_scales, torch::kFloat32),
         ratios = torch::tensor(_anchor_ratios, torch::kFloat32),
         strides = torch::tensor(_feat_stride, torch::kFloat32);
    // [num_strides, num_scales, 1]
    scales = (strides.view({-1, 1}) * scales).unsqueeze(-1);
    // [num_strides, num_scales, num_ratios]
    auto w = scales / ratios.sqrt(), h = scales * ratios.sqrt();
    // [num_strides, num_scales, num_ratios, 4], center at upper-left corner
    _base_anchors = torch::stack({-w / 2, -h / 2, w / 2, h / 2}, -1);
    // _base_anchors += strides.view({-1, 1, 1, 1}) * _center_offset; // move center by offset
    std::cout << "_base_anchors:\n" << _base_anchors << std::endl;
    register_buffer("base_anchors", _base_anchors); // device follows module's device
}

torch::Tensor AnchorGeneratorImpl::get_anchors(const std::vector<int64_t> &grid_size)
{
    auto stride = _feat_stride;
    auto base_anchor = _base_anchors; // [num_scales, num_ratios, 4]
    auto grid_x = grid_size[1], grid_y = grid_size[0];
    auto x_offset = torch::arange(grid_x, base_anchor.device()) * stride; // [grid_x]
    auto y_offset = torch::arange(grid_y, base_anchor.device()) * stride; // [grid_y]

    auto anchors = base_anchor.repeat({grid_y, grid_x, 1, 1, 1}); // [grid_y, grid_x, num_scales, num_ratios, 4]
    auto shape = anchors.sizes().vec();
    // from [grid_y, grid_x, num_scales, num_ratios, 4]
    // to   [grid_y, grid_x, num_scales, num_ratios, 2, 2]
    // so that we can add offset to the last dim
    shape.pop_back();
    shape.push_back(2);
    shape.push_back(2); // can't find an easier way
    anchors = anchors.view(shape);
    // move x by x_offset and y by y_offset
    anchors.index_put_({"...", 1}, y_offset.view({-1, 1, 1, 1, 1}) + anchors.index({"...", 1}));
    anchors.index_put_({"...", 0}, x_offset.view({-1, 1, 1, 1}) + anchors.index({"...", 0}));
    // from [grid_y, grid_x, num_scales, num_ratios, 2, 2]
    // to   [grid_y, grid_x, num_scales, num_ratios, 4]
    shape.pop_back();
    shape.pop_back();
    shape.push_back(4);
    anchors = anchors.view(shape);
    // the shape of anchor for current stride is [grid_y, grid_x, num_scales, num_ratios, 4]
    return anchors;
}

int AnchorGeneratorImpl::num_anchors()
{
    return _anchor_scales.size() * _anchor_ratios.size();
}

} // namespace anchor
