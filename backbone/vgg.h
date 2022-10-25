#pragma once

#include <torch/torch.h>

namespace backbone
{
/// this is a special case vgg16D model, for reusing in faster rcnn model
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

const std::vector<int> cfg_D = {
    64, 64, -1 /* stands for M */, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};

torch::nn::Sequential make_layers(const std::vector<int> &cfg, bool batch_norm = false);

VGG make_vgg16(int num_classes, bool pretrained = true);

} // namespace backbone
