#pragma once

#include "vision_cuda.h"

inline at::Tensor nms(
    const at::Tensor &dets,
    const at::Tensor &scores,
    const double iou_threshold)
{
    assert(dets.device().is_cuda() && "CUDA is required");
    if (dets.numel() == 0)
    {
        at::cuda::CUDAGuard device_guard(dets.device());
        return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return nms_cuda(dets, scores, iou_threshold);
}
