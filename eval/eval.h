#pragma once

#include <map>
#include <string_view>
#include <tuple>
#include <vector>

namespace eval
{

constexpr std::array<std::string_view, 20> voc_categories = {
    "aeroplane" /*0*/,    "bicycle" /*1*/, "bird" /*2*/,   "boat" /*3*/,       "bottle" /*4*/,
    "bus" /*5*/,          "car" /*6*/,     "cat" /*7*/,    "chair" /*8*/,      "cow" /*9*/,
    "diningtable" /*10*/, "dog" /*11*/,    "horse" /*12*/, "motorbike" /*13*/, "person" /*14*/,
    "pottedplant" /*15*/, "sheep" /*16*/,  "sofa" /*17*/,  "train" /*18*/,     "tvmonitor" /*19*/};

// Compute VOC AP given precision and recall.
// If use_07_metric is true, uses the
// VOC 07 11 point method (default: true).
float VOC_ap(const std::valarray<float> &recall, const std::valarray<float> &precision, bool use_07_metric = false);

// return precision, recall, ap for given class
std::tuple<float, float, float> VOCEval(std::string_view result_path, std::string_view anno_path,
                                        std::string_view imageset_path, std::string_view class_name,
                                        float thresh = 0.5f, bool use_07_metric = true);

// return precision, recall, ap for all given class
std::tuple<const std::array<float, voc_categories.size()>, const std::array<float, voc_categories.size()>,
           const std::array<float, voc_categories.size()>>
VOCEval(std::string_view result_path, std::string_view anno_path, std::string_view imageset_path,
        const std::array<std::string_view, voc_categories.size()> &class_names, float thresh = 0.5f,
        bool use_07_metric = true);
} // namespace eval