#pragma once

#include "example.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <torch/data/transforms.h>

namespace dataset
{

// A `Transform` that applies a user-provided functor to individual examples.
class ResizeTransform : public torch::data::transforms::Transform<DetectionExample, DetectionExample>
{
  public:
    using typename torch::data::transforms::Transform<DetectionExample, DetectionExample>::InputType;
    using typename torch::data::transforms::Transform<DetectionExample, DetectionExample>::OutputType;

    /// Constructs the `ResizeTransform` from the given parameters.
    explicit ResizeTransform(std::vector<float> data_size) : data_size(std::move(data_size))
    {
    }

    /// Resizes both the image and the gt_bboxes
    OutputType apply(InputType input) override
    {
        cv::Mat img = input.data;
        auto h = img.rows;
        auto w = img.cols;
        auto max_side = std::max(data_size.at(0), data_size.at(1));
        auto min_side = std::min(data_size.at(0), data_size.at(1));
        auto scale_factor = std::min(std::min(max_side / h, max_side / w), std::max(min_side / h, min_side / w));

        // resize image
        cv::Mat rescaled;
        cv::resize(img, rescaled, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        input.data = rescaled;

        // resize gt_bboxes
        input.target.first = input.target.first * scale_factor;
        input.img_shape = std::vector<int64_t>({rescaled.rows, rescaled.cols});
        input.scale_factor = scale_factor;
    }

  private:
    std::vector<float> data_size;
};
} // namespace dataset