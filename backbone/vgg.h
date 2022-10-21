#pragma once

#include <torch/torch.h>

namespace backbone
{

struct VGGImpl : public torch::nn::Module
{
    VGGImpl(torch::nn::Sequential features, int num_classes = 1000, bool init_weights = true);
    torch::Tensor forward(torch::Tensor x);

    void _initialize_weights();
    torch::nn::Sequential features;
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier;
};

TORCH_MODULE(VGG);

torch::nn::Sequential make_layers(const std::vector<int> &cfg, bool batch_norm = false);

VGG make_vgg16(int num_classes, bool pretrained = true);

} // namespace backbone
