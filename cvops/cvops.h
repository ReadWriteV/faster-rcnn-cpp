#pragma once

#ifndef WITH_CUDA
#error "CUDA is required"
#endif

#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
