#pragma once

#include <torch/torch.h>

#include <string>

namespace dataset
{

/// A `DetectionExample` from a dataset.
///
/// A detection dataset consists of data and an associated target (annotation).
struct DetectionExample
{
    struct Target
    {
        Target() = default;
        Target(torch::Tensor gt_bboxes, torch::Tensor gt_labels)
            : gt_bboxes(std::move(gt_bboxes)), gt_labels(std::move(gt_labels))
        {
        }
        Target &to(const torch::TensorOptions &opts)
        {
            gt_bboxes = gt_bboxes.to(opts);
            gt_labels = gt_labels.to(opts);
            return *this;
        }
        torch::Tensor gt_bboxes;
        torch::Tensor gt_labels;
    };
    using DataType = torch::Tensor;
    using TargetType = Target;

    DetectionExample() = default;
    DetectionExample(DataType data, TargetType target) : data(std::move(data)), target(std::move(target))
    {
    }

    void to(const torch::TensorOptions &opts)
    {
        data = data.to(opts);
        target = target.to(opts);
    }

    DataType data;
    TargetType target;
    std::vector<int64_t> img_shape;
    float scale_factor{-1.0};
    std::string id;
};
} // namespace dataset