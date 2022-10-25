#include "vgg.h"
#include <memory>
#include <stdexcept>
#include <torch/nn/init.h>

namespace backbone
{

VGGImpl::VGGImpl(int num_classes, bool init_weights)
{
    const std::array cfg_D = {
        64, 64, -1 /* stands for M */, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};

    int in_channels = 3;
    int count = 0;
    for (auto v : cfg_D)
    {
        count++;
        if (v == -1) // stands for M
        {
            this->features->push_back("MaxPool2d" + std::to_string(count),
                                      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        }
        else
        {
            this->features->push_back("Conv2d" + std::to_string(count),
                                      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, v, 3).padding(1)));
            this->features->push_back("ReLU" + std::to_string(count), torch::nn::ReLU(torch::nn::ReLUOptions(true)));
            in_channels = v;
        }
    }

    this->avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));

    this->classifier->push_back("Linear1", torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    this->classifier->push_back("ReLU2", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    this->classifier->push_back("Dropout3", torch::nn::Dropout());
    this->classifier->push_back("Linear4", torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    this->classifier->push_back("ReLU5", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    this->classifier->push_back("Dropout6", torch::nn::Dropout());
    this->classifier->push_back("Linear7", torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));

    features = register_module("features", features);
    avgpool = register_module("avgpool", avgpool);
    classifier = register_module("classifier", classifier);

    if (init_weights)
    {
        _initialize_weights();
    }
}

void VGGImpl::_initialize_weights()
{
    throw std::logic_error("Pretrained model should not initialize weights");
}

torch::Tensor VGGImpl::forward(torch::Tensor x)
{
    x = features->forward(x);
    x = avgpool->forward(x);
    x = x.flatten(1);
    x = classifier->forward(x);
    return x;
}
} // namespace backbone
