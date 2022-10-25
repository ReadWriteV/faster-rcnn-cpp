#include "vgg.h"
#include <memory>
#include <stdexcept>
#include <torch/nn/init.h>

namespace backbone
{

VGGImpl::VGGImpl(torch::nn::Sequential features, int num_classes, bool init_weights)
{

    this->features = features;

    this->avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));

    this->classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    this->classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    this->classifier->push_back(torch::nn::Dropout());
    this->classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    this->classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    this->classifier->push_back(torch::nn::Dropout());
    this->classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));

    features = register_module("features", features);
    avgpool = register_module("avgpool", avgpool);
    classifier = register_module("classifier", classifier);

    if (init_weights)
    {
        _initialize_weights();
    }
}

torch::Tensor VGGImpl::forward(torch::Tensor x)
{
    x = features->forward(x);
    x = avgpool->forward(x);
    x = x.flatten(1);
    x = classifier->forward(x);
    return x;
}

void VGGImpl::_initialize_weights()
{
    throw std::logic_error("Pretrained model should not initialize weights");
    for (auto &m : modules(/* include_self set to  */ false))
    {
        if (auto M = dynamic_cast<torch::nn::Conv2dImpl *>(m.get()))
        {
            torch::nn::init::kaiming_normal_(M->weight, 0, torch::kFanOut, torch::kReLU);
            if (M->bias.defined())
            {
                torch::nn::init::constant_(M->bias, 0);
            }
        }
        else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl *>(m.get()))
        {
            torch::nn::init::constant_(M->weight, 1);
            torch::nn::init::constant_(M->bias, 0);
        }
        else if (auto M = dynamic_cast<torch::nn::LinearImpl *>(m.get()))
        {
            torch::nn::init::normal_(M->weight, 0, 0.01);
            torch::nn::init::constant_(M->bias, 0);
        }
    }
}

torch::nn::Sequential make_layers(const std::vector<int> &cfg, bool batch_norm)
{
    torch::nn::Sequential layers;
    int in_channels = 3;
    for (auto v : cfg)
    {
        if (v == -1) // stands for M
        {
            layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        }
        else
        {
            layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, v, 3).padding(1)));
            if (batch_norm)
            {
                layers->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v)));
            }
            layers->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
            in_channels = v;
        }
    }
    return layers;
}

VGG make_vgg16(int num_classes, bool pretrained)
{
    return VGG(make_layers(cfg_D, false), num_classes, !pretrained);
}

} // namespace backbone
