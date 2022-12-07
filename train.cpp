#include <boost/json/stream_parser.hpp>
#include <boost/json/value.hpp>
#include <boost/program_options.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "detector.h"
#include "voc.h"

int main(int argc, char **argv)
{
    std::string config_file_path;
    int gpu_id = 0;
    try
    {
        boost::program_options::options_description train_options_desc("Model training options");
        // clang-format off
        train_options_desc.add_options()
        ("help,h", "help guide")
        ("path,p", boost::program_options::value(&config_file_path)->default_value("./config/faster_rcnn_vgg16.json"), "config file path")
        ("gpu,g", boost::program_options::value(&gpu_id)->default_value(0), "id of gpu");
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
        std::ifstream config_file;
        config_file.open(config_file_path);
        assert(config_file.is_open());
        boost::json::stream_parser p;
        boost::json::error_code ec;
        long nread = 0;
        do
        {
            char buf[4096];
            nread = config_file.readsome(buf, sizeof(buf));
            p.write(buf, nread, ec);
        } while (nread != 0);
        if (ec)
        {
            return -1;
        }
        p.finish(ec);
        if (ec)
        {
            return -1;
        }
        const auto cfg = p.release();

        std::cout << "use gpu: " << gpu_id << std::endl;

        torch::Device _device(torch::kCUDA, static_cast<torch::DeviceIndex>(gpu_id));

        auto dataset = std::make_unique<dataset::VOCDataset>(cfg.at_pointer("/data/dataset_path").as_string().c_str(),
                                                             dataset::Mode::train);
        std::cout << "train size: " << dataset->size().value() << std::endl;

        auto model = detector::FasterRCNNVGG16(cfg.at("model"));

        const auto &optimizer_opts = cfg.at("optimizer");
        assert(optimizer_opts.at("type") == "SGD" && "only support SGD optimizer");

        // construct SGD options for later construction of SGD optimizer
        auto optim_opts = torch::optim::SGDOptions(optimizer_opts.at("lr").as_double())
                              .momentum(optimizer_opts.at("momentum").as_double())
                              .weight_decay(optimizer_opts.at("weight_decay").as_double());
        double epoch_lr = optimizer_opts.at("lr").as_double();

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

        const auto &lr_opts = cfg.at("lr_opts");

        const auto &decay_epochs = lr_opts.at("decay_epochs").as_array();

        const auto total_epochs = cfg.at("total_epochs").as_int64();
        const auto save_ckpt_period = cfg.at("save_ckpt_period").as_int64();
        const auto log_period = cfg.at("log_period").as_int64();
        const auto &work_dir = cfg.at("work_dir").as_string();

        std::filesystem::path save_folder_path("output");
        save_folder_path /= work_dir.c_str();
        if (std::filesystem::exists(save_folder_path) == false)
        {
            std::filesystem::create_directory(save_folder_path);
        }

        // Train
        utils::ProgressTracker pg_tracker(total_epochs, dataset->size().value()); // track train process

        model->to(_device);
        model->train();
        std::cout << model << std::endl;
        auto loader_opts =
            torch::data::DataLoaderOptions().batch_size(1).workers(cfg.at_pointer("/data/train_workers").as_int64());
        auto dataloader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(*dataset), loader_opts);
        // start to train model epoch by epoch
        for (int64_t epoch = 1; epoch <= total_epochs; epoch++)
        {
            // check if lr needs to be decayed
            if (std::find(decay_epochs.begin(), decay_epochs.end(), epoch) != decay_epochs.end())
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