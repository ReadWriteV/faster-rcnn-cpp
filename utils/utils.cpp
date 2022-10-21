#include "utils.h"

#include <cassert>

namespace utils
{
using namespace torch::indexing;

torch::Tensor xyxy2xywh(torch::Tensor xyxy)
{
    return torch::stack({(xyxy.index({Slice(), 0}) + xyxy.index({Slice(), 2})) * 0.5,
                         (xyxy.index({Slice(), 1}) + xyxy.index({Slice(), 3})) * 0.5,
                         xyxy.index({Slice(), 2}) - xyxy.index({Slice(), 0}),
                         xyxy.index({Slice(), 3}) - xyxy.index({Slice(), 1})},
                        -1);
}

torch::Tensor xywh2xyxy(torch::Tensor xywh)
{
    const auto ctr_xy = xywh.index({Slice(), Slice(None, 2)});
    const auto half_wh = xywh.index({Slice(), Slice(2, None)}) * 0.5;
    return torch::cat({ctr_xy - half_wh, ctr_xy + half_wh}, -1);
}

torch::Tensor xyxy2xywhcoco(torch::Tensor xyxy)
{
    return torch::stack({xyxy.index({Slice(), 0}), xyxy.index({Slice(), 1}),
                         xyxy.index({Slice(), 2}) - xyxy.index({Slice(), 0}),
                         xyxy.index({Slice(), 3}) - xyxy.index({Slice(), 1})},
                        -1);
}

torch::Tensor bbox_area(torch::Tensor bboxes)
{
    const auto xywh = xyxy2xywh(bboxes);
    return xywh.index({Slice(), 2}) * xywh.index({Slice(), 3});
}

torch::Tensor rand_choice(torch::Tensor tsr, int num, int dim)
{
    assert((dim < tsr.dim() && num <= tsr.size(dim)) && "invalid input for rand_choice");
    const auto chosen_index =
        torch::randperm(tsr.size(dim), torch::TensorOptions().dtype(torch::kLong).device(tsr.device()))
            .index({Slice(None, num)});
    return tsr.index_select(dim, chosen_index);
}

std::vector<std::vector<int64_t>> get_grid_size(const std::vector<torch::Tensor> &feats)
{
    std::vector<std::vector<int64_t>> sizes;
    for (const auto &feat : feats)
    {
        auto vec = feat.sizes().vec();
        sizes.emplace_back(vec.begin() + 2, vec.begin() + 4);
    }
    return sizes;
}

std::vector<torch::Tensor> batch_reshape(std::vector<torch::Tensor> &tensors, const std::vector<int64_t> &size)
{
    std::vector<torch::Tensor> reshaped;
    for (auto &tsr : tensors)
    {
        reshaped.push_back(tsr.reshape(size));
    }
    return reshaped;
}

std::vector<torch::Tensor> batch_permute(std::vector<torch::Tensor> &tensors, const std::vector<int64_t> &size)
{
    std::vector<torch::Tensor> permute;
    for (auto &tsr : tensors)
    {
        permute.push_back(tsr.permute(size));
    }
    return permute;
}

std::vector<torch::Tensor> batch_repeat(std::vector<torch::Tensor> &tensors, const std::vector<int64_t> &size)
{
    std::vector<torch::Tensor> rpt;
    for (auto &tsr : tensors)
    {
        rpt.push_back(tsr.repeat(size));
    }
    return rpt;
}

torch::Tensor restrict_bbox(torch::Tensor bboxes, const std::vector<int64_t> &max_shape)
{
    const auto &max_h = max_shape.at(0), &max_w = max_shape.at(1);
    return torch::stack(
        {
            bboxes.index({Slice(), 0}).clamp(0, max_w),
            bboxes.index({Slice(), 1}).clamp(0, max_h),
            bboxes.index({Slice(), 2}).clamp(0, max_w),
            bboxes.index({Slice(), 3}).clamp(0, max_h),
        },
        1);
}

std::string ProgressTracker::secs2str(int64_t secs)
{
    std::string s_str;
    int64_t n = 0;
    if ((n = secs / 86400) > 0)
    {
        s_str += std::to_string(n) + "d";
        secs = secs - n * 86400;
    } // days
    if ((n = secs / 3600) > 0)
    {
        s_str += std::to_string(n) + "h";
        secs = secs - n * 3600;
    } // hours
    if ((n = secs / 60) > 0)
    {
        s_str += std::to_string(n) + "m";
        secs = secs - n * 60;
    } // minutes
    if ((n = secs / 1) > 0)
    {
        s_str += std::to_string(n) + "s";
        secs = secs - n * 1;
    } // secs
    if (s_str.size() == 0)
        return "0s";
    return s_str;
}

std::string ProgressTracker::now_str()
{
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string now_str(50, '\0');
    std::strftime(&now_str[0], now_str.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(now_str.c_str());
}

double ProgressTracker::elapsed()
{
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _start).count();
}

double ProgressTracker::eta()
{
    auto _elapsed = elapsed();
    return _elapsed / (_cur_iter + 1) * (_total_iters - _cur_iter);
}

double ProgressTracker::fps()
{
    return _cur_iter / elapsed();
}

void ProgressTracker::progress_bar()
{
    int bar_size = 50;
    int progress = std::min(int((double)cur_iter() / total_iters() * bar_size + 0.5), bar_size);
    std::cout.flush();
    std::cout << "[" << std::string(progress, '#') << std::string(bar_size - progress, '.') << "]"
              << ", elapsed: " << secs2str(elapsed()) << ", eta: " << secs2str(eta()) << ", fps: " << std::fixed
              << std::setprecision(2) << fps() << "\r";
    std::cout.flush();
}

void ProgressTracker::track_loss(const std::map<std::string, torch::Tensor> &losses)
{
    for (const auto &loss : losses)
    {
        _tracked_loss[loss.first].push_back(loss.second.item<double>());
    }
}

std::map<std::string, double> ProgressTracker::mean_loss(bool clear_history)
{
    std::map<std::string, double> report;
    for (auto &loss : _tracked_loss)
    {
        if (loss.second.size() == 0)
        {
            return report;
        }
        report[loss.first] = std::accumulate(loss.second.begin(), loss.second.end(), 0.0) / loss.second.size();
    }
    if (clear_history)
    {
        for (auto &loss : _tracked_loss)
        {
            loss.second.clear();
        }
    }
    return report;
}

void ProgressTracker::report_progress(std::ostream &os)
{
    auto loss_report = mean_loss();
    os << now_str() << ", epoch[" << cur_epoch() << "][" << cur_iter() % iters_per_epoch() << "/" << iters_per_epoch()
       << "], lr: " << std::fixed << std::setprecision(5) << lr() << ", ";
    for (auto &loss_key : {"rpn_cls_loss", "rpn_bbox_loss", "rcnn_cls_loss", "rcnn_bbox_loss", "loss"})
    {
        std::cout << loss_key << ":" << std::fixed << std::setprecision(3) << loss_report[loss_key] << ", ";
    }
    os << "eta: " << secs2str(eta()) << ", fps: " << fps();
    os << std::endl;
}
} // namespace utils
