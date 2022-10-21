#include "eval.h"

#include <algorithm>
#include <array>
#include <boost/algorithm/string.hpp>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <valarray>
#include <vector>

#include <boost/property_tree/ptree_fwd.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace eval
{

std::size_t get_category_id(std::string_view name)
{
    return std::distance(voc_categories.begin(), std::find(voc_categories.begin(), voc_categories.end(), name));
}

float VOC_ap(const std::valarray<float> &recall, const std::valarray<float> &precision, bool use_07_metric)
{
    assert(recall.size() == precision.size());
    double ap = 0.0;
    if (use_07_metric)
    {
        // 11 point metric
        for (double t = 0.0; t <= 1.0; t += 0.1) // 0.0 0.1 0.2 ... 0.9 1.0
        {
            float p = 0.0;
            for (std::size_t i = 0; i < recall.size(); i++) // p = np.max(prec[rec >= t])
            {
                if (recall[i] >= t)
                {
                    p = std::max(precision[i], p);
                }
            }
            ap += p / 11.0;
        }
    }
    else
    {
        // correct AP calculation
        std::vector<float> m_recall, m_precision;
        m_recall.reserve(recall.size() + 2);
        m_precision.reserve(precision.size() + 2);

        // first append sentinel values at the end
        m_recall.push_back(0.0);
        m_recall.insert(m_recall.end(), std::begin(recall), std::end(recall));
        m_recall.push_back(1.0);

        m_precision.push_back(0.0);
        m_precision.insert(m_precision.end(), std::begin(precision), std::end(precision));
        m_precision.push_back(0.0);

        // compute the precision envelope
        for (std::size_t i = m_precision.size() - 1; i > 0; i--)
        {
            m_precision.at(i - 1) = std::max(m_precision.at(i - 1), m_precision.at(i));
        }

        // to calculate area under PR curve, look for points
        // where X axis (recall) changes value
        std::vector<std::size_t> indexs;
        for (std::size_t i = 0; i < m_recall.size() - 1; i++)
        {
            if (m_recall.at(i) != m_recall.at(i + 1))
            {
                indexs.push_back(i);
            }
        }

        // and sum (\Delta recall) * prec
        for (const auto i : indexs)
        {
            ap += (m_recall.at(i + 1) - m_recall.at(i)) * m_precision.at(i + 1);
        }
    }
    return ap;
}

std::vector<std::tuple<std::array<int, 4>, int, bool>> get_anno(std::string_view anno_file_name)
{
    std::vector<std::tuple<std::array<int, 4>, int, bool>> annos;

    boost::property_tree::ptree pt;
    boost::property_tree::read_xml(anno_file_name.data(), pt);
    auto objects = pt.get_child("annotation");
    for (const auto &object : objects)
    {
        if (object.first == "object")
        {
            auto class_name = object.second.get<std::string>("name");
            int gt_label = get_category_id(class_name);
            bool difficult = static_cast<bool>(object.second.get<int>("difficult"));
            std::array<int, 4> gt_bbox = {object.second.get<int>("bndbox.xmin"), object.second.get<int>("bndbox.ymin"),
                                          object.second.get<int>("bndbox.xmax"), object.second.get<int>("bndbox.ymax")};

            annos.emplace_back(gt_bbox, gt_label, difficult);
        }
    }
    return annos;
}

std::tuple<float, float, float> VOCEval(std::string_view result_path, std::string_view anno_path,
                                        std::string_view imageset_path, std::string_view class_name, float thresh,
                                        bool use_07_metric)
{
    // read imageset file
    std::ifstream imageset_file(imageset_path.data());
    if (imageset_file.fail())
    {
        throw std::runtime_error("open imageset_file failed");
    }

    std::vector<std::string> imageset;
    std::string image_index;
    std::map<std::string, std::vector<std::tuple<std::array<int, 4>, int, bool>>> annos; // GT annos
    // load GT from dataset
    while (std::getline(imageset_file, image_index))
    {
        imageset.push_back(image_index);
        annos[image_index] = get_anno(std::string(anno_path) + "/" + image_index + ".xml");
    }
    int class_id = get_category_id(class_name);
    std::map<std::string, std::tuple<std::vector<std::array<int, 4>>, std::vector<bool>, std::vector<bool>>>
        class_annos; // GT for class_name, 0: bbox, 1: difficult, 2: det
    std::vector<bool> det;
    std::size_t npos = 0;
    for (const auto &imagename : imageset)
    {
        std::vector<std::array<int, 4>> R;
        std::vector<bool> difficult;
        for (const auto &e : annos.at(imagename))
        {
            if (std::get<1>(e) == class_id)
            {
                R.push_back(std::get<0>(e));
                difficult.push_back(std::get<2>(e));
                if (std::get<2>(e) == false)
                {
                    npos++;
                }
            }
        }
        std::vector<bool> det(R.size(), false);
        class_annos.insert(
            std::make_pair(imagename, std::make_tuple(std::move(R), std::move(difficult), std::move(det))));
    }

    // read detfile
    std::ifstream result_file(result_path.data());
    if (result_file.fail())
    {
        throw std::runtime_error("open result_file failed");
    }

    std::vector<std::tuple<std::string, float, std::array<float, 4>>> results; // image_ids, confidence, BB
    // load detection result from detfile
    std::string line;
    while (std::getline(result_file, line))
    {
        std::vector<std::string> splitlines;
        boost::split(splitlines, line, boost::is_any_of(" "), boost::token_compress_on);
        assert(splitlines.size() == 6);
        results.emplace_back(splitlines[0], std::stof(splitlines[1]),
                             std::array{std::stof(splitlines[2]), std::stof(splitlines[3]), std::stof(splitlines[4]),
                                        std::stof(splitlines[5])});
    }
    std::size_t nd = results.size();
    std::valarray<float> tp(0.0f, nd);
    std::valarray<float> fp(0.0f, nd);
    std::valarray<float> fn(0.0f, nd);

    if (results.size() > 0)
    {
        // sort by confidence descend
        std::sort(results.begin(), results.end(),
                  [](const std::tuple<std::string, float, std::array<float, 4>> &a,
                     const std::tuple<std::string, float, std::array<float, 4>> b) {
                      return std::get<1>(a) > std::get<1>(b);
                  });
        // go down dets and mark TPs and FPs
        for (std::size_t i = 0; i < nd; i++)
        {
            auto &R = class_annos[std::get<0>(results.at(i))]; // GT annos
            auto bb = std::get<2>(results.at(i));              // detected bbox
            float ovmax = std::numeric_limits<float>::lowest();
            std::size_t jmax;
            auto BBGT = std::get<0>(R); // GT bboxs

            if (BBGT.size() > 0)
            {
                // compute overlaps
                // intersection, union
                std::valarray<float> inters(0.0f, BBGT.size()), uni(0.0f, BBGT.size()), overlaps(0.0f, BBGT.size());

                std::for_each(BBGT.begin(), BBGT.end(),
                              [&bb, &inters, &uni, index = 0ULL](const std::array<int, 4> &e) mutable {
                                  float ixmin = std::max<float>(e[0], bb[0]);
                                  float iymin = std::max<float>(e[1], bb[1]);
                                  float ixmax = std::min<float>(e[2], bb[2]);
                                  float iymax = std::min<float>(e[3], bb[3]);
                                  float iw = std::max(ixmax - ixmin + 1, 0.0f);
                                  float ih = std::max(iymax - iymin + 1, 0.0f);
                                  float inter = iw * ih;
                                  inters[index] = inter;
                                  float bb_area = (bb[2] - bb[0] + 1.0f) * (bb[3] - bb[1] + 1.0f);
                                  float e_area = (e[2] - e[0] + 1.0f) * (e[3] - e[1] + 1.0f);
                                  uni[index] = bb_area + e_area - inter;
                                  index++;
                              });
                overlaps = inters / uni;
                auto max_iter = std::max_element(std::begin(overlaps), std::end(overlaps));
                ovmax = *max_iter;
                jmax = std::distance(std::begin(overlaps), max_iter);
            }
            if (ovmax > thresh)
            {
                if (std::get<1>(R)[jmax] == false)
                {
                    if (std::get<2>(R)[jmax] == false)
                    {
                        tp[i] = 1.0f;
                        std::get<2>(R)[jmax] = true;
                    }
                    else
                    {
                        fp[i] = 1.0f;
                    }
                }
            }
            else
            {
                fp[i] = 1.0f;
            }
        }
    }

    for (std::size_t i = 1; i < nd; i++)
    {
        tp[i] += tp[i - 1];
        fp[i] += fp[i - 1];
    }

    fn = static_cast<float>(npos) - tp;
    std::valarray<float> rec = tp / static_cast<float>(npos);

    // avoid divide by zero in case the first detection matches a difficult ground truth
    std::valarray<float> prec =
        tp / (tp + fp).apply([](float e) -> float { return std::max(e, std::numeric_limits<float>::min()); });

    auto ap = VOC_ap(rec, prec, use_07_metric);

    return {prec[prec.size() - 1], rec[rec.size() - 1], ap};
}

std::tuple<const std::array<float, voc_categories.size()>, const std::array<float, voc_categories.size()>,
           const std::array<float, voc_categories.size()>>
VOCEval(std::string_view result_path, std::string_view anno_path, std::string_view imageset_path,
        const std::array<std::string_view, voc_categories.size()> &class_names, float thresh, bool use_07_metric)
{
    return {};
}
} // namespace eval