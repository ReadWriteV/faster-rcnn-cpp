#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <torch/torch.h>

#include "example.h"

namespace dataset
{

/// The mode in which the dataset is loaded.
enum class Mode
{
    train,
    test
};

/// @brief fetch annotation, read image and does transforms
class VOCDataset : public torch::data::datasets::Dataset<VOCDataset, DetectionExample>
{
  public:
    VOCDataset(const std::filesystem::path &root, Mode mode = Mode::train, bool non_difficult = true);

    /// @brief fetch all data of idx-th image
    /// @param index index of image
    /// @return all data
    DetectionExample get(size_t index) override;

    /// Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    constexpr static std::string_view get_category_name(std::size_t i);
    static std::size_t get_category_id(std::string_view i);

    static constexpr std::size_t num_class = 20;
    static constexpr std::array<std::string_view, num_class> categories = {
        "aeroplane" /*0*/, "bicycle", "bird",        "boat",     "bottle", "bus",       "car",    "cat",
        "chair",           "cow",     "diningtable", "dog",      "horse",  "motorbike", "person", "pottedplant",
        "sheep",           "sofa",    "train",       "tvmonitor"};
    static std::map<std::string_view, std::size_t> categories_name_to_id; // map from categories_name to id

  private:
    void transform(ExampleType &img_data, cv::Mat &image_data); // TODO: use map method

    std::filesystem::path root;
    std::vector<std::string> examples_index;                          // map from image index to image name
    std::map<std::size_t, DetectionExample::TargetType> example_anns; // map from image index to gt_bboxes and gt_labels

    Mode mode;          // train or test
    bool non_difficult; // wh non-difficult bboxes are used
};

// the following defines common image transforms like rescale and flip etc.
std::pair<cv::Mat, float> rescale_image(cv::Mat img, std::vector<float> img_scale);
cv::Mat flip_image(cv::Mat img, const std::string &dire);
torch::Tensor flip_bboxes(torch::Tensor bboxes, std::vector<int64_t> img_shape, const std::string &dire);
cv::Mat normalize_image(cv::Mat img, std::vector<float> mean, std::vector<float> std);
cv::Mat pad_image(cv::Mat img, int divisor);

} // namespace dataset