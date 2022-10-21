#pragma once

#include "vision_cuda.h"

// Interface for Python
inline torch::Tensor ROIAlign_forward(
    const torch::Tensor &input, // Input feature map.
    const torch::Tensor &rois,  // List of ROIs to pool over.
    const double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    const int64_t pooled_height,  // The height of the pooled feature map.
    const int64_t pooled_width,   // The width of the pooled feature
    const int64_t sampling_ratio, // The number of points to sample in each bin
    const bool aligned)           // The flag for pixel shift
// along each axis.
{
    assert(input.device().is_cuda() && "CUDA is required");
    return ROIAlign_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned);
}

inline torch::Tensor ROIAlign_backward(
    const torch::Tensor &grad,
    const torch::Tensor &rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    const bool aligned)
{
    assert(grad.device().is_cuda() && "CUDA is required");

    return ROIAlign_backward_cuda(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio,
        aligned);
}

class ROIAlignFunction : public torch::autograd::Function<ROIAlignFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::Variable input,
        torch::autograd::Variable rois,
        const double spatial_scale,
        const int64_t pooled_height,
        const int64_t pooled_width,
        const int64_t sampling_ratio,
        const bool aligned)
    {
        ctx->saved_data["spatial_scale"] = spatial_scale;
        ctx->saved_data["pooled_height"] = pooled_height;
        ctx->saved_data["pooled_width"] = pooled_width;
        ctx->saved_data["sampling_ratio"] = sampling_ratio;
        ctx->saved_data["aligned"] = aligned;
        ctx->saved_data["input_shape"] = input.sizes();
        ctx->save_for_backward({rois});
        auto result = ROIAlign_forward(
            input,
            rois,
            spatial_scale,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned);
        return {result};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output)
    {
        // Use data saved in forward
        auto saved = ctx->get_saved_variables();
        auto rois = saved[0];
        auto input_shape = ctx->saved_data["input_shape"].toIntList();
        auto grad_in = ROIAlign_backward(
            grad_output[0],
            rois,
            ctx->saved_data["spatial_scale"].toDouble(),
            ctx->saved_data["pooled_height"].toInt(),
            ctx->saved_data["pooled_width"].toInt(),
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            ctx->saved_data["sampling_ratio"].toInt(),
            ctx->saved_data["aligned"].toBool());
        return {grad_in,
                torch::autograd::Variable(),
                torch::autograd::Variable(),
                torch::autograd::Variable(),
                torch::autograd::Variable(),
                torch::autograd::Variable(),
                torch::autograd::Variable()};
    }
};

inline torch::Tensor roi_align(
    const torch::Tensor &input,
    const torch::Tensor &rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio,
    const bool aligned)
{
    return ROIAlignFunction::apply(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned)[0];
}
