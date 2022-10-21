#pragma once

#include <boost/property_tree/ptree.hpp>
#include <torch/torch.h>

namespace neck
{
/// @brief A simple implementation of FPN
class FPNImpl : public torch::nn::Module
{
  public:
    FPNImpl(int out_channels = 256, const std::vector<int> &feat_channels = {256, 512, 1024, 2048}, int num_outs = 5);
    FPNImpl(const boost::property_tree::ptree &opts);

    void init();

    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> feats);

  private:
    int _out_channels;
    std::vector<int> _feat_channels;
    int _num_outs;

    torch::nn::ModuleList _lateral_convs{nullptr};
    torch::nn::ModuleList _fpn_convs{nullptr};

    void init_weights();
};
TORCH_MODULE(FPN);

} // namespace neck
