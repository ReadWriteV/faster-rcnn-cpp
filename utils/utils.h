#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace utils
{
/// @brief convert [x_min, y_min, x_max, y_max] coordinate format to [center_x, center_y, width, height]
/// @param xyxy [x_min, y_min, x_max, y_max]
/// @return [center_x, center_y, width, height], (center_x, center_y): center point, (width, height): width and height
torch::Tensor xyxy2xywh(torch::Tensor xyxy);

/// @brief convert [center_x, center_y, width, height] coordinate format to [x_min, y_min, x_max, y_max]
/// @param xywh [center_x, center_y, width, height]
/// @return [x_min, y_min, x_max, y_max], (x_min, y_min): top-left point, (x_max, y_max): bottom-right point
torch::Tensor xywh2xyxy(torch::Tensor xywh);

/// @brief convert [x_min, y_min, x_max, y_max] coordinate format to [x_min, y_min, width, height]
/// @param xywh [x_min, y_min, x_max, y_max]
/// @return [x_min, y_min, width, height], (x_min, y_min): top-left point, (width, height): width and height
torch::Tensor xyxy2xywhcoco(torch::Tensor xyxy);

/// @brief caculate area of given bounding boxes
/// @param bboxes given bounding boxes
/// @return area of given bounding boxes
torch::Tensor bbox_area(torch::Tensor bboxes);

/// @brief randomly select 'num' from tsr's 'dim' dimension without replacement,
/// which means 'num' should be <= tsr.size(dim). it uses torch.randperm which
/// is sub-optimal but the choice without using numpy
/// @param tsr input tensor
/// @param num number to  select
/// @param dim target dimension
/// @return selected tensor
torch::Tensor rand_choice(torch::Tensor tsr, int num, int dim = 0);

/// @brief get w and h of given features map list
/// @param feats given features map list
/// @return vector of w and h
std::vector<std::vector<int64_t>> get_grid_size(const std::vector<torch::Tensor> &feats);

std::vector<torch::Tensor> batch_reshape(std::vector<torch::Tensor> &tensors, const std::vector<int64_t> &size);

std::vector<torch::Tensor> batch_permute(std::vector<torch::Tensor> &tensors, const std::vector<int64_t> &size);

std::vector<torch::Tensor> batch_repeat(std::vector<torch::Tensor> &tensors, const std::vector<int64_t> &size);

/// @brief clamp bbox in (0, 0) ~ (max_w, max_h)
/// @param bboxes bounding boxes
/// @param max_shape (max_w, max_h)
/// @return precessed bounding boxes
torch::Tensor restrict_bbox(torch::Tensor bboxes, const std::vector<int64_t> &max_shape);

/// @brief It helps to track progress of training and and testing.
/// For training, it tracks lr, losses, eta, current iteration and epoch.
/// For testing, it prints a progress bar.
class ProgressTracker
{
  public:
    static std::string now_str();
    static std::string secs2str(int64_t secs);
    ProgressTracker(int64_t epochs, int64_t iters_per_epoch)
        : _total_epochs(epochs), _iters_per_epoch(iters_per_epoch), _lr(0)
    {
        _cur_iter = 0;
        _cur_epoch = 1;
        _total_iters = epochs * iters_per_epoch;
        _start = std::chrono::high_resolution_clock::now();
    }

    void next_iter()
    {
        _cur_iter++;
    }
    void next_epoch()
    {
        _cur_epoch++;
    }
    double elapsed();
    double eta();
    double fps();
    double lr()
    {
        return _lr;
    }

    void track_loss(const std::map<std::string, torch::Tensor> &losses);
    void track_lr(double lr)
    {
        _lr = lr;
    }
    std::map<std::string, double> mean_loss(bool clear_history = true); // report mean loss

    void progress_bar(); // print progress bar
    void report_progress(std::ostream &os);

    int64_t total_epochs()
    {
        return _total_epochs;
    }
    int64_t iters_per_epoch()
    {
        return _iters_per_epoch;
    }
    int64_t cur_iter()
    {
        return _cur_iter;
    }
    int64_t cur_epoch()
    {
        return _cur_epoch;
    }
    int64_t total_iters()
    {
        return _total_iters;
    }

  private:
    std::map<std::string, std::vector<double>> _tracked_loss;
    double _lr;
    int64_t _total_epochs;
    int64_t _iters_per_epoch;
    int64_t _cur_iter;
    int64_t _cur_epoch;
    int64_t _total_iters;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};

} // namespace utils