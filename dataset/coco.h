#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <map>
#include <opencv2/core/mat.hpp>
#include <string>
#include <torch/data/example.h>
#include <utility>
#include <vector>

#include <ATen/core/TensorBody.h>
#include <opencv2/opencv.hpp>
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
class COCODataset : public torch::data::datasets::Dataset<COCODataset, DetectionExample>
{
  public:
    COCODataset(const std::filesystem::path &image_path, const std::filesystem::path &annotation_path,
                Mode mode = Mode::train);

    /// @brief fetch all data of idx-th image
    /// @param index index of image
    /// @return all data
    DetectionExample get(size_t index) override;

    /// Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    static constexpr std::size_t num_class = 20;

    std::string get_category_name(std::size_t i) const;

  private:
    void transform(ExampleType &img_data, cv::Mat &image_data); // TODO: use map method

    std::filesystem::path image_path;
    std::filesystem::path annotation_file_path;

    std::vector<std::string> examples_index;                  // map from image index to image name (image id)
    std::map<std::string, std::size_t> categories_name_to_id; // map from categories_name to id
    std::array<std::string, num_class> categories;
    std::map<int, DetectionExample::TargetType> example_anns; // map from image id to gt_bboxes and gt_labels
    Mode mode;                                                // train or test
};

// the following defines common image transforms like rescale and flip etc.
std::pair<cv::Mat, float> rescale_image(cv::Mat img, std::vector<float> img_scale);
cv::Mat flip_image(cv::Mat img, const std::string &dire);
torch::Tensor flip_bboxes(torch::Tensor bboxes, std::vector<int64_t> img_shape, const std::string &dire);
cv::Mat normalize_image(cv::Mat img, std::vector<float> mean, std::vector<float> std);
cv::Mat pad_image(cv::Mat img, int divisor);

} // namespace dataset