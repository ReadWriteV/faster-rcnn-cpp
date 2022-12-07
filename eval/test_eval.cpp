#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <valarray>

#include "eval.h"

#include <boost/program_options.hpp>

int main(int argc, char **argv)
{
    std::string result_path;
    std::string anno_path;
    std::string imageset_path;
    int gpu_id = -1;
    int resume_epoch = -1;
    try
    {
        boost::program_options::options_description train_options_desc("VOC Eval options");
        // clang-format off
        train_options_desc.add_options()
        ("help,h", "help guide")
        ("result_path,r", boost::program_options::value(&result_path)->required(), "result file path")
        ("anno_path,a", boost::program_options::value(&anno_path)->default_value("/mnt/889cdd89-1094-48ae-b221-146ffe543605/wr/datasets/VOCdevkit/VOC2007/Annotations"), "annos path of VOC")
        ("imageset_path,i", boost::program_options::value(&imageset_path)->default_value("/mnt/889cdd89-1094-48ae-b221-146ffe543605/wr/datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt"), "test.txt path of VOC");
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

    float map = 0;
    for (auto class_name : eval::categories)
    {
        std::string result_file = std::string(result_path) + std::string(class_name) + ".txt";
        auto [prec, rec, ap] = eval::VOCEval(result_file, anno_path, imageset_path, class_name);
        std::cout << std::left << std::setw(12) << class_name << ": AP = " << std::setprecision(4) << std::fixed << ap
                  << ", PRE = " << prec << ", REC = " << rec << std::endl;
        map += ap;
    }
    std::cout << "mAP = " << map / eval::categories.size() << std::endl;
    return 0;
}