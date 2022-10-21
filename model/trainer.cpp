#include "trainer.h"

#include <c10/core/Device.h>
#include <cassert>
#include <iterator>
#include <string>
#include <torch/data/samplers/random.h>

namespace trainer
{
BasicTrainer::BasicTrainer(const boost::property_tree::ptree &pt)
    : _pg_tracker(0, 0), _opts(pt), _device(torch::kCUDA, static_cast<torch::DeviceIndex>(pt.get("gpu", -1)))
{
    // construct VOC dataset
    _dataset = std::make_shared<dataset::VOCDataset>(_opts.get<std::string>("data.dataset_path"), dataset::Mode::train);

    // construct COCO dataset
    // _dataset = std::make_shared<dataset::COCODataset>(_opts.get<std::string>("data.dataset_image_path"),
    //                                                  _opts.get<std::string>("data.train_dataset_annotation_path"),
    //                                                  dataset::Mode::train);

    auto model_opts = _opts.get_child("model");
    // construct FasterRCNN object detector
    _model = detector::FasterRCNN(model_opts.get_child("backbone"), model_opts.get_child("neck"),
                                  model_opts.get_child("rpn_head"), model_opts.get_child("rcnn_head"));

    auto optimizer_opts = _opts.get_child("optimizer");
    assert(optimizer_opts.get<std::string>("type") == "SGD" && "only support SGD optimizer");
    // construct SGD options for later construction of SGD optimizer
    auto optim_opts = torch::optim::SGDOptions(optimizer_opts.get<float>("lr"))
                          .momentum(optimizer_opts.get<float>("momentum"))
                          .weight_decay(optimizer_opts.get<float>("weight_decay"));
    _epoch_lr = optimizer_opts.get<float>("lr");
    // construct SGD optimizer
    _optimizer = std::make_shared<torch::optim::SGD>(_model->parameters(), optim_opts);

    std::cout << "[SGDOptions] lr: " << optim_opts.lr() << ", momentum: " << optim_opts.momentum()
              << ", weight_decay: " << optim_opts.weight_decay() << std::endl;

    auto lr_opts = _opts.get_child("lr_opts");

    const auto &decay_epochs_node = lr_opts.get_child("decay_epochs");
    std::transform(decay_epochs_node.begin(), decay_epochs_node.end(),
                   std::inserter(_decay_epochs, _decay_epochs.begin()),
                   [](const boost::property_tree::ptree::value_type &v) { return v.second.get_value<int>(); });
    _warmup_start = lr_opts.get<float>("warmup_start");
    _warmup_steps = lr_opts.get<float>("warmup_steps");

    std::cout << "warmup_start: " << _warmup_start << std::endl;
    std::cout << "warmup_steps: " << _warmup_steps << std::endl;
    std::cout << "decay_epochs: " << _decay_epochs << std::endl;

    _total_epochs = _opts.get<int>("total_epochs");
    _save_ckpt_period = _opts.get<int>("save_ckpt_period");
    _log_period = _opts.get<int>("log_period");
    _work_dir = _opts.get("work_dir", "work_dir");
}

void BasicTrainer::train()
{
    _pg_tracker = utils::ProgressTracker(_total_epochs, _dataset->size().value());
    _model->to(_device);
    _model->train();
    std::cout << _model << std::endl;
    auto loader_opts = torch::data::DataLoaderOptions().batch_size(1).workers(_opts.get<int>("data.train_workers"));
    auto dataloader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(*_dataset), loader_opts);
    // start to train model epoch by epoch
    for (int64_t epoch = 1; epoch <= _total_epochs; epoch++)
    {
        // check if lr needs to be decayed
        if (_decay_epochs.find(epoch) != _decay_epochs.end())
        {
            _epoch_lr *= 0.1;
            set_lr(_epoch_lr);
        }

        // iterate over all image data
        for (auto &img_datas : *dataloader)
        {
            // check if lr needs to be warmed up at the begining
            auto img_data = img_datas[0];
            img_data.to(_device);
            warmup_lr();
            auto model_loss = _model->forward_train(img_data);
            auto tot_loss = sum_loss(model_loss);
            model_loss["loss"] = tot_loss;
            _pg_tracker.track_loss(model_loss);
            _optimizer->zero_grad();
            tot_loss.backward();
            _optimizer->step();
            _pg_tracker.next_iter();
            if (_pg_tracker.cur_iter() % _log_period == 0)
            {
                _pg_tracker.track_lr(get_lr());
                _pg_tracker.report_progress(std::cout);
            }
        }
        _pg_tracker.next_epoch();
        if (epoch % _save_ckpt_period == 0)
        {
            // _work_dir must exist
            torch::save(_model, _work_dir + "/epoch_" + std::to_string(epoch) + ".pt");
        }
    }
}

void BasicTrainer::warmup_lr()
{
    auto iters = _pg_tracker.cur_iter();
    if (iters <= _warmup_steps)
    {
        float lr = _warmup_start * _epoch_lr + (1 - _warmup_start) * iters / _warmup_steps * _epoch_lr;
        set_lr(lr);
    }
}

void BasicTrainer::set_lr(float lr)
{
    for (auto &group : _optimizer->param_groups())
    {
        static_cast<torch::optim::SGDOptions &>(group.options()).lr(lr);
    }
}

float BasicTrainer::get_lr()
{
    return static_cast<torch::optim::SGDOptions &>(_optimizer->param_groups()[0].options()).lr();
}

torch::Tensor BasicTrainer::sum_loss(const std::map<std::string, torch::Tensor> &loss_map)
{
    auto tot_loss = torch::tensor(0, torch::TensorOptions().dtype(torch::kFloat32).device(_device));
    for (auto &loss : loss_map)
    {
        tot_loss += loss.second;
    }
    return tot_loss;
}

} // namespace trainer
