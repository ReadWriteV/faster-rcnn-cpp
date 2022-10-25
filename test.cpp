#include "detector.h"
#include "voc.h"

#include <array>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <stdexcept>
#include <torch/torch.h>

int main(int argc, char **argv)
{

    std::string config_file_path;
    std::string model_file_path;
    std::string result_folder_path;
    int gpu_id;

    try
    {
        boost::program_options::options_description train_options_desc("Model testing options");
        // clang-format off
        train_options_desc.add_options()
        ("help,h", "help guide")
        ("path,p", boost::program_options::value<std::string>(&config_file_path)->default_value("./config/faster_rcnn_vgg16.json"), "config file path")
        ("model,m", boost::program_options::value<std::string>(&model_file_path)->required(), "model file path")
        ("result_path,r", boost::program_options::value<std::string>(&result_folder_path)->required(), "result file save folder")
        ("gpu,g", boost::program_options::value(&gpu_id)->default_value(-1), "id of gpu");
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

        auto model_opts = opts.get_child("model");
        // construct FasterRCNN object detector
        auto model = detector::FasterRCNNVGG16(model_opts.get_child("backbone"), model_opts.get_child("rpn_head"),
                                               model_opts.get_child("rcnn_head"));
        auto device = torch::Device(torch::kCUDA, gpu_id);
        torch::load(model, model_file_path);
        model->eval();
        model->to(device);
        auto dataset = std::make_shared<dataset::VOCDataset>(opts.get<std::string>("data.dataset_path"),
                                                             dataset::Mode::test, true);

        utils::ProgressTracker pg_tracker(1, dataset->size().value());
        auto loader_opts = torch::data::DataLoaderOptions().batch_size(1).workers(opts.get<int>("data.test_workers"));
        auto dataloader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(*dataset), loader_opts);

        torch::NoGradGuard no_grad;

        std::filesystem::path result_folder(result_folder_path);
        if (std::filesystem::exists(result_folder) == false)
        {
            std::filesystem::create_directory(result_folder);
        }
        std::array<std::ofstream, dataset::VOCDataset::num_class> result_files;
        for (std::size_t i = 0; i < result_files.size(); i++)
        {
            result_files.at(i).open(result_folder / (std::string(dataset::VOCDataset::categories.at(i)) + ".txt"));
            if (result_files.at(i).fail())
            {
                throw std::runtime_error("create result file failed");
            }
        }

        for (auto &batch : *dataloader)
        {
            auto example = batch[0];
            example.to(device);
            auto det_res = model->forward_test(example);
            auto det_bboxes = std::get<0>(det_res), det_scores = std::get<1>(det_res),
                 det_labels = std::get<2>(det_res);
            // det_bboxes = utils::xyxy2xywhcoco(det_bboxes) / example.scale_factor;
            det_bboxes = det_bboxes / example.scale_factor;
            for (int i = 0; i < det_bboxes.size(0); i++)
            {
                auto label = det_labels[i].item<long>();
                auto score = det_scores[i].item<float>();
                auto bbox = det_bboxes[i];
                result_files.at(label) << std::setw(6) << std::setfill('0') << example.id << ' ';
                result_files.at(label) << score << ' ';
                result_files.at(label) << bbox[0].item<float>() << ' ' << bbox[1].item<float>() << ' '
                                       << bbox[2].item<float>() << ' ' << bbox[3].item<float>() << '\n';
            };

            pg_tracker.next_iter();
            if (pg_tracker.cur_iter() % 100 == 0)
            {
                pg_tracker.progress_bar();
            }
            if (pg_tracker.cur_iter() == pg_tracker.total_iters())
            {
                pg_tracker.progress_bar();
            }
        }
        std::cout << "write results to file..." << std::endl;
        std::cout << "done" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
    return 0;
}
