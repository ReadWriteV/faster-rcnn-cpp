#pragma once

#include "detector.h"
#include "voc.h"

#include <boost/property_tree/ptree.hpp>

namespace trainer
{
/*
  It constructs a model and then trains the model. It prints out basic training information like:
  timestamp, epoch#, iter/tot_iter, lr, eta, losses.
**/
class BasicTrainer
{
  public:
    BasicTrainer(const boost::property_tree::ptree &pt);
    void train();

  private:
    void warmup_lr();
    void set_lr(float lr);
    float get_lr();
    torch::Tensor sum_loss(const std::map<std::string, torch::Tensor> &loss_map);

    // private members
    detector::FasterRCNN _model{nullptr};
    std::shared_ptr<dataset::VOCDataset> _dataset{nullptr};
    // std::shared_ptr<dataset::COCODataset> _dataset{nullptr};

    std::shared_ptr<torch::optim::SGD> _optimizer{nullptr};

    std::set<int> _decay_epochs;
    float _warmup_start;
    float _warmup_steps;
    int _total_epochs;
    int _save_ckpt_period;
    int _log_period;

    const boost::property_tree::ptree &_opts;

    torch::Device _device;
    std::string _work_dir;
    float _epoch_lr{-1};
    utils::ProgressTracker _pg_tracker; // track train process
};

} // namespace trainer
