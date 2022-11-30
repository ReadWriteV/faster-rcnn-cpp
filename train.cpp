#include <algorithm>
#include <filesystem>
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "detector.h"
#include "voc.h"

int main(int argc, char **argv)
{
    std::string config_file_path;
    int gpu_id = -1;
    int resume_epoch = -1;
    try
    {
        boost::program_options::options_description train_options_desc("Model training options");
        // clang-format off
        train_options_desc.add_options()
        ("help,h", "help guide")
        ("path,p", boost::program_options::value(&config_file_path)->default_value("./config/faster_rcnn_vgg16.json"), "config file path")
        ("resume,r", boost::program_options::value(&resume_epoch), "resume training from given epoch")
        ("gpu,g", boost::program_options::value(&gpu_id), "id of gpu");
        // clang-format on

        boost::program_options::variables_map vm;

        // if (argc < 2)
        // {
        //     std::cerr << train_options_desc << std::endl;
        //     return -1;
        // }
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, train_options_desc), vm);

        if (vm.count("help") > 0)
        {
            std::cout << train_options_desc << std::endl;
            return -1;
        }
        boost::program_options::notify(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }

    try
    {
        if (std::filesystem::exists(config_file_path) == false)
        {
            std::cerr << config_file_path << " NOT exist, check path!" << '\n';
            return -1;
        }
        boost::property_tree::ptree opts;
        boost::property_tree::read_json(config_file_path, opts);
        if (gpu_id != -1)
        {
            opts.put("gpu", gpu_id);
            std::cout << "use gpu: " << opts.get("gpu", -1) << std::endl;
        }
        if (resume_epoch != -1)
        {
            opts.put("resume", resume_epoch);
            std::cout << "resume from epoch " << opts.get("resume", 0) << std::endl;
        }

        torch::Device _device(torch::kCUDA, static_cast<torch::DeviceIndex>(opts.get("gpu", -1)));

        auto dataset =
            std::make_unique<dataset::VOCDataset>(opts.get<std::string>("data.dataset_path"), dataset::Mode::train);
        std::cout << "train size: " << dataset->size().value() << std::endl;

        auto model_opts = opts.get_child("model");
        auto model = detector::FasterRCNNVGG16(model_opts.get_child("backbone"), model_opts.get_child("rpn_head"),
                                               model_opts.get_child("rcnn_head"));

        auto optimizer_opts = opts.get_child("optimizer");
        assert(optimizer_opts.get<std::string>("type") == "SGD" && "only support SGD optimizer");

        // construct SGD options for later construction of SGD optimizer
        auto optim_opts = torch::optim::SGDOptions(optimizer_opts.get<float>("lr"))
                              .momentum(optimizer_opts.get<float>("momentum"))
                              .weight_decay(optimizer_opts.get<float>("weight_decay"));
        float epoch_lr = optimizer_opts.get<float>("lr");

        // construct SGD optimizer
        // std::vector<torch::Tensor> params;
        // for (auto &e : model->parameters())
        // {
        //     if (e.requires_grad())
        //     {
        //         params.push_back(e);
        //     }
        // }

        auto optimizer = std::make_unique<torch::optim::SGD>(model->parameters(), optim_opts);

        std::cout << "[SGDOptions] lr: " << optim_opts.lr() << ", momentum: " << optim_opts.momentum()
                  << ", weight_decay: " << optim_opts.weight_decay() << std::endl;

        auto lr_opts = opts.get_child("lr_opts");

        std::set<int> decay_epochs;
        const auto &decay_epochs_node = lr_opts.get_child("decay_epochs");
        std::transform(decay_epochs_node.begin(), decay_epochs_node.end(),
                       std::inserter(decay_epochs, decay_epochs.begin()),
                       [](const boost::property_tree::ptree::value_type &v) { return v.second.get_value<int>(); });

        int total_epochs = opts.get<int>("total_epochs");
        int save_ckpt_period = opts.get<int>("save_ckpt_period");
        int log_period = opts.get<int>("log_period");
        std::string work_dir = opts.get("work_dir", "work_dir");

        std::filesystem::path save_folder_path("output");
        save_folder_path /= work_dir;
        if (std::filesystem::exists(save_folder_path) == false)
        {
            std::filesystem::create_directory(save_folder_path);
        }

        // Train
        utils::ProgressTracker pg_tracker(total_epochs, dataset->size().value()); // track train process

        model->to(_device);
        model->train();
        std::cout << model << std::endl;
        auto loader_opts = torch::data::DataLoaderOptions().batch_size(1).workers(opts.get<int>("data.train_workers"));
        auto dataloader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(*dataset), loader_opts);
        // start to train model epoch by epoch
        for (int64_t epoch = 1; epoch <= total_epochs; epoch++)
        {
            // check if lr needs to be decayed
            if (decay_epochs.find(epoch) != decay_epochs.end())
            {
                epoch_lr *= 0.1;
                for (auto &group : optimizer->param_groups())
                {
                    static_cast<torch::optim::SGDOptions &>(group.options()).lr(epoch_lr);
                }
            }

            // iterate over all image data
            for (auto &img_datas : *dataloader) // No stack
            {
                // check if lr needs to be warmed up at the begining
                auto img_data = img_datas[0];
                img_data.to(_device);
                model->zero_grad();
                auto model_loss = model->forward_train(img_data);
                auto tot_loss = torch::tensor(0, torch::TensorOptions().dtype(torch::kFloat32).device(_device));
                for (const auto &loss : model_loss)
                {
                    tot_loss += loss.second;
                }
                model_loss["loss"] = tot_loss;

                pg_tracker.track_loss(model_loss);
                optimizer->zero_grad();
                tot_loss.backward();
                optimizer->step();
                pg_tracker.next_iter();
                if (pg_tracker.cur_iter() % log_period == 0)
                {
                    pg_tracker.track_lr(
                        static_cast<torch::optim::SGDOptions &>(optimizer->param_groups()[0].options()).lr());
                    pg_tracker.report_progress(std::cout);
                }
            }
            pg_tracker.next_epoch();
            if (epoch % save_ckpt_period == 0)
            {
                std::string save_file = "epoch_" + std::to_string(epoch) + ".pt";
                torch::save(model, save_folder_path / save_file);
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
    return 0;
}