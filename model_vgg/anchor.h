#pragma once

#include "utils.h"
#include <torch/torch.h>

#include <boost/json/value.hpp>

namespace anchor
{
class AnchorGeneratorImpl : public torch::nn::Module
{
  public:
    // Center_offset used to be 0.5, i.e. center of base anchors is at the center of grid.
    // Later people tend to let center_offset=0, i.e. base anchors are centered at upper-left of the grid.
    AnchorGeneratorImpl(std::vector<float> anchor_scales = {8.0f, 16.f, 32.0f},
                        std::vector<float> anchor_ratios = {0.5f, 1.0f, 2.0f}, int feat_stride = 16);

    AnchorGeneratorImpl(const boost::json::value &opts);

    /**
      Need two steps to generate anchors:
      1, generate base_anchors.
      2, tile base_anchors along x-axis and y-axis.
       Generate base_anchors:
         1, scale = stride * scale
         2, w*h = scale^2
      3, h/w = ratio
     */
    torch::Tensor get_anchors(const std::vector<int64_t> &grid_size);
    // number of base anchors
    int num_anchors();

  private:
    std::vector<float> _anchor_scales;
    std::vector<float> _anchor_ratios;
    int _feat_stride;

    torch::Tensor _base_anchors;

}; // class AnchorGeneratorImpl
TORCH_MODULE(AnchorGenerator);

} // namespace anchor
