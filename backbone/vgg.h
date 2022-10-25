#pragma once

#include <torch/torch.h>

namespace backbone
{
/// this is a special case vgg16D model, for reusing in faster rcnn model
struct VGGImpl : public torch::nn::Module
{
    VGGImpl(int num_classes = 1000, bool init_weights = true);
    torch::Tensor forward(torch::Tensor x);

    void _initialize_weights();

    torch::nn::Sequential features;
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier;
};

TORCH_MODULE(VGG);

} // namespace backbone
