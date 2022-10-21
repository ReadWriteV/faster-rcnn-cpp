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
#include <torch/data/transforms/tensor.h>

#include "trainer.h"

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
        ("path,p", boost::program_options::value(&config_file_path)->default_value("./config/faster_rcnn_r50_fpn_1x_voc.json"), "config file path")
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
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(config_file_path, pt);
        if (gpu_id != -1)
        {
            pt.put("gpu", gpu_id);
            std::cout << "use gpu: " << pt.get("gpu", -1) << std::endl;
        }
        if (resume_epoch != -1)
        {
            pt.put("resume", resume_epoch);
            std::cout << "resume from epoch " << pt.get("resume", 0) << std::endl;
        }
        auto trainer = trainer::BasicTrainer(pt);
        trainer.train();
    }

    catch (std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
    return 0;
}