#pragma once

#include <map>
#include <string_view>
#include <tuple>
#include <vector>

namespace eval
{

static constexpr std::array categories = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"};

// Compute VOC AP given precision and recall.
// If use_07_metric is true, uses the
// VOC 07 11 point method (default: true).
float VOC_ap(const std::valarray<float> &recall, const std::valarray<float> &precision, bool use_07_metric = false);

// return precision, recall, ap for given class
std::tuple<float, float, float> VOCEval(std::string_view result_path, std::string_view anno_path,
                                        std::string_view imageset_path, std::string_view class_name,
                                        float thresh = 0.5f, bool use_07_metric = true);

// return precision, recall, ap for all given class
std::tuple<const std::array<float, categories.size()>, const std::array<float, categories.size()>,
           const std::array<float, categories.size()>>
VOCEval(std::string_view result_path, std::string_view anno_path, std::string_view imageset_path,
        const std::array<std::string_view, categories.size()> &class_names, float thresh = 0.5f,
        bool use_07_metric = true);
} // namespace eval