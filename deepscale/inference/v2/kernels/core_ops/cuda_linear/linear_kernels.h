// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

#ifndef DEEPSCALE_CUDA_LINEAR_KERNELS_H
#define DEEPSCALE_CUDA_LINEAR_KERNELS_H

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "ds_kernel_utils.h"

#include "linear_kernels_cuda.h"

void cuda_wf6af16_linear(torch::Tensor& output,
                         torch::Tensor& hidden_states,
                         torch::Tensor& weights_2bit,
                         torch::Tensor& weights_4bit,
                         torch::Tensor& scale,
                         torch::Tensor& workspace,
                         int M,
                         int N,
                         int K,
                         int split_k);

std::vector<torch::Tensor> preprocess_weight(torch::Tensor& Weight);

#endif
