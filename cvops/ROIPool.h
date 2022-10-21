#pragma once

#include "vision_cuda.h"

inline std::tuple<at::Tensor, at::Tensor> ROIPool_forward(
    const at::Tensor &input,
    const at::Tensor &rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width)
{
    assert(input.device().is_cuda() && "CUDA is required");

    return ROIPool_forward_cuda(
        input, rois, spatial_scale, pooled_height, pooled_width);
}

inline at::Tensor ROIPool_backward(
    const at::Tensor &grad,
    const at::Tensor &rois,
    const at::Tensor &argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width)
{
    assert(grad.device().is_cuda() && "CUDA is required");
    return ROIPool_backward_cuda(
        grad,
        rois,
        argmax,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width);
}

class ROIPoolFunction : public torch::autograd::Function<ROIPoolFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::Variable input,
        torch::autograd::Variable rois,
        const double spatial_scale,
        const int64_t pooled_height,
        const int64_t pooled_width)
    {
        ctx->saved_data["spatial_scale"] = spatial_scale;
        ctx->saved_data["pooled_height"] = pooled_height;
        ctx->saved_data["pooled_width"] = pooled_width;
        ctx->saved_data["input_shape"] = input.sizes();
        auto result = ROIPool_forward(
            input, rois, spatial_scale, pooled_height, pooled_width);
        auto output = std::get<0>(result);
        auto argmax = std::get<1>(result);
        ctx->save_for_backward({rois, argmax});
        ctx->mark_non_differentiable({argmax});
        return {output, argmax};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output)
    {
        // Use data saved in forward
        auto saved = ctx->get_saved_variables();
        auto rois = saved[0];
        auto argmax = saved[1];
        auto input_shape = ctx->saved_data["input_shape"].toIntList();
        auto grad_in = ROIPool_backward(
            grad_output[0],
            rois,
            argmax,
            ctx->saved_data["spatial_scale"].toDouble(),
            ctx->saved_data["pooled_height"].toInt(),
            ctx->saved_data["pooled_width"].toInt(),
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3]);
        return {grad_in, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }
};

inline std::tuple<torch::Tensor, torch::Tensor> roi_pool(
    const torch::Tensor &input,
    const torch::Tensor &rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width)
{
    auto result = ROIPoolFunction::apply(
        input, rois, spatial_scale, pooled_height, pooled_width);
    return std::tuple<torch::Tensor, torch::Tensor>(result[0], result[1]);
}
