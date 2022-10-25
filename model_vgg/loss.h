#pragma once

#include "bbox.h"

#include <boost/property_tree/ptree_fwd.hpp>
#include <string>
#include <torch/torch.h>

namespace loss
{
// loss base class
class Loss : public torch::nn::Module
{
  public:
    virtual torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor)
    {
        throw std::runtime_error("Please implement loss");
    }
};

// L1 loss
class L1Loss : public Loss
{
  public:
    L1Loss(double loss_weight);
    L1Loss(const boost::property_tree::ptree &opts);

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;

  private:
    double _loss_weight;
};

// GIoU loss
class GIoULoss : public Loss
{
  public:
    GIoULoss(double loss_weight);
    GIoULoss(const boost::property_tree::ptree &opts);

    // assume both pred and target represent a same amount of bboxes
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;

  private:
    double _loss_weight;
};

// CE loss
class CrossEntropyLoss : public Loss
{
  public:
    CrossEntropyLoss(double loss_weight);
    CrossEntropyLoss(const boost::property_tree::ptree &opts);

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;

  private:
    double _loss_weight;
};

// BCE loss
class BinaryCrossEntropyLoss : public Loss
{
  public:
    BinaryCrossEntropyLoss(double loss_weight);
    BinaryCrossEntropyLoss(const boost::property_tree::ptree &opts);

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;

  private:
    double _loss_weight;
};

std::shared_ptr<Loss> build_loss(const boost::property_tree::ptree &opts);

} // namespace loss
