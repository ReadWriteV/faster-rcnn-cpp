#include "voc.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <torch/data/transforms/tensor.h>
#include <utility>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace dataset
{

std::unordered_map<std::string_view, std::size_t> VOCDataset::categories_name_to_id{};

VOCDataset::VOCDataset(const std::filesystem::path &root, Mode mode, bool non_difficult)
    : root(root), mode(mode), non_difficult(non_difficult)
{
    if (std::filesystem::exists(root) == false)
    {
        throw std::runtime_error(root.string() + ": bad dataset path!");
    }
    std::cout << "load VOC dataset from " << root << ", Mode: " << ((mode == Mode::train) ? "train" : "test")
              << std::endl;

    // map from 0 get better performance then from 1
    std::transform(categories.begin(), categories.end(),
                   std::inserter(categories_name_to_id, categories_name_to_id.begin()),
                   [i = 0ULL](std::string_view e) mutable { return std::make_pair(e, i++); });
    // TODO: configurable ImageSets file
    auto index_file_path = root / "ImageSets" / "Main" / (mode == Mode::train ? "trainval.txt" : "test.txt");
    auto annotation_path = root / "Annotations";

    std::ifstream file(index_file_path);
    if (file.fail())
    {
        throw std::runtime_error("open " + index_file_path.string() + " failed");
    }

    std::string example_index;
    std::size_t i = 0;
    while (std::getline(file, example_index))
    {

        boost::property_tree::ptree pt;
        boost::property_tree::read_xml(annotation_path / (example_index + ".xml"), pt);
        std::vector<torch::Tensor> gt_bboxes, gt_labels;
        auto objects = pt.get_child("annotation");
        for (const auto &object : objects)
        {
            if (object.first == "object")
            {
                if (non_difficult && object.second.get<int>("difficult") > 0)
                {
                    continue;
                }
                if (categories_name_to_id.find(object.second.get<std::string>("name")) == categories_name_to_id.end())
                {
                    continue;
                }
                std::vector<int> bbox{object.second.get<int>("bndbox.xmin"), object.second.get<int>("bndbox.ymin"),
                                      object.second.get<int>("bndbox.xmax"), object.second.get<int>("bndbox.ymax")};
                gt_bboxes.push_back(torch::tensor(bbox));
                gt_labels.push_back(torch::tensor(static_cast<int64_t>(
                                                      categories_name_to_id.at(object.second.get<std::string>("name"))))
                                        .view(1));
            }
        }
        int64_t num = gt_bboxes.size();
        if (num == 0)
        {
            continue;
        }
        examples_index.push_back(example_index);
        example_anns[i].gt_bboxes = torch::cat(gt_bboxes).view({num, 4});
        example_anns[i].gt_labels = torch::cat(gt_labels);
        i++;
    }
    // assert(examples_index.size() == (mode == Mode::train ? 2975 : 500));
    std::cout << "total examples loaded: " << examples_index.size() << std::endl;
}

DetectionExample VOCDataset::get(size_t index)
{
    ExampleType example;
    auto image_file_path = root / "JPEGImages" / (examples_index.at(index) + ".jpg");

    auto image_mat = cv::imread(image_file_path, cv::IMREAD_COLOR);
    if (image_mat.empty())
    {
        throw std::runtime_error("can not read image: " + image_file_path.string());
    }

    // idata.ori_shape = std::vector<int64_t>{img_cv2.rows, img_cv2.cols, 3};

    // set gt_bboxes and gt_labels
    example.target = example_anns.at(index);
    example.id = examples_index.at(index);
    // apply tansform
    transform(example, image_mat);

    // cv::Mat img = image_mat;
    // img.convertTo(img, CV_32F);
    // example.data = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
    // example.data = example.data.permute({2, 0, 1}).unsqueeze(0).contiguous();

    return example;
}

torch::optional<size_t> VOCDataset::size() const
{
    return examples_index.size();
}

constexpr std::string_view VOCDataset::get_category_name(std::size_t i)
{
    return categories.at(i);
}

std::size_t VOCDataset::get_category_id(std::string_view i)
{
    return categories_name_to_id.at(i);
}

void VOCDataset::transform(ExampleType &example, cv::Mat &image_data)
{
    cv::Mat img = image_data;
    // do not forget to transform gt_bboxes
    // resize
    std::vector<float> scale_range{1000.0f, 600.0f};
    auto scaled = rescale_image(img, scale_range);
    img = scaled.first;
    auto scale_factor = scaled.second;

    // resize gt_bboxes
    example.target.gt_bboxes = example.target.gt_bboxes * scale_factor;
    example.img_shape = std::vector<int64_t>({img.rows, img.cols});
    example.scale_factor = scale_factor;

    // flip
    if (mode == Mode::train)
    {
        float flip_ratio = 0.5f;
        // same random sequence every run
        static std::default_random_engine eng{};
        static std::uniform_real_distribution<float> distr{0.0f, 1.0f};
        if (distr(eng) < flip_ratio)
        {
            std::vector<int64_t> img_shape({img.rows, img.cols});
            img = flip_image(img, "horizontal");
            example.target.gt_bboxes = flip_bboxes(example.target.gt_bboxes, img_shape, "horizontal");
        }
    }

    // convert to float32
    img.convertTo(img, CV_32F);

    // normalize
    std::vector<float> mean = {123.675f, 116.28f, 103.53f};
    std::vector<float> std = {58.395f, 57.12f, 57.375f};
    img = normalize_image(img, mean, std);

    // pad
    int pad_divisor = 32;
    img = pad_image(img, pad_divisor);
    // img_data.pad_shape = std::vector<int64_t>({img.rows, img.cols});

    // cv2::mat to torch tensor
    example.data = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
    example.data = example.data.permute({2, 0, 1}).unsqueeze(0).contiguous(); // unsqueeze(0), so size is [1, c, h, w]
}

std::pair<cv::Mat, float> rescale_image(cv::Mat img, std::vector<float> img_scale)
{
    auto h = img.rows;
    auto w = img.cols;
    auto max_side = std::max(img_scale[0], img_scale[1]);
    auto min_side = std::min(img_scale[0], img_scale[1]);
    auto scale = std::min(std::min(max_side / h, max_side / w), std::max(min_side / h, min_side / w));
    cv::Mat rescaled;
    cv::resize(img, rescaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
    return std::make_pair(rescaled, scale);
}

cv::Mat flip_image(cv::Mat img, const std::string &dire)
{
    if (dire == "horizontal")
    {
        cv::flip(img, img, 1);
    }
    else if (dire == "vertical")
    {
        cv::flip(img, img, 0);
    }
    else
    {
        throw std::runtime_error("unknown flip direction: " + dire);
    }
    return img;
}

torch::Tensor flip_bboxes(torch::Tensor bboxes, std::vector<int64_t> img_shape, const std::string &dire)
{
    auto h = img_shape[0];
    auto w = img_shape[1];
    if (dire == "horizontal")
    {
        auto flipped_x =
            w -
            torch::stack({bboxes.index({torch::indexing::Slice(), 2}), bboxes.index({torch::indexing::Slice(), 0})}, 1);
        bboxes.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 4, 2)}, flipped_x);
    }
    else if (dire == "vertical")
    {
        auto flipped_y =
            h -
            torch::stack({bboxes.index({torch::indexing::Slice(), 3}), bboxes.index({torch::indexing::Slice(), 1})}, 1);
        bboxes.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, 4, 2)}, flipped_y);
    }
    else
    {
        throw std::runtime_error("unknown flip direction: " + dire);
    }
    return bboxes;
}

// mean and std are all in order [R, G, B], so may need to reorder color channels first
cv::Mat normalize_image(cv::Mat img, std::vector<float> mean, std::vector<float> std)
{
    // convert from BGR to RGB, assume img is from direct result of cv::imread
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    auto scalar_mean = cv::Scalar({mean[0], mean[1], mean[2]});
    auto scalar_std = cv::Scalar({std[0], std[1], std[2]});
    cv::subtract(img, scalar_mean, img);
    cv::divide(img, scalar_std, img);
    return img;
}

cv::Mat pad_image(cv::Mat img, int divisor)
{
    assert(divisor > 0 && "divisor must be positive");
    int tar_h = ((img.rows - 1) / divisor + 1) * divisor;
    int tar_w = ((img.cols - 1) / divisor + 1) * divisor;
    cv::copyMakeBorder(img, img, 0, tar_h - img.rows, 0, tar_w - img.cols, cv::BORDER_CONSTANT, 0);
    return img;
}
} // namespace dataset
