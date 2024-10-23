// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "bias_activation.h"
#include "blas.h"
#include "gated_activation_kernels.h"
#include "layer_norm.h"
#include "linear_kernels.h"
#include "rms_norm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // bias_activation.h
    m.def("bias_activation", &bias_activation, "DeepScale bias activation in CUDA");

    // layer_norm.h
    m.def("layer_norm", &ds_layer_norm, "DeepScale layer norm in CUDA");
    m.def("pre_layer_norm", &ds_pre_layer_norm, "DeepScale pre layer norm in CUDA");
    m.def("post_layer_norm", &ds_post_layer_norm, "DeepScale pre layer norm in CUDA");

    // blas.h
    m.def("blas_linear", &blas_linear, "Linear implemented by vendor BLAS");
    m.def("blas_4d_matmul", &blas_4d_matmul, "4D matmul implemented by vendor BLAS");
    m.def("create_handle", &create_handle, "Create a handle for vendor BLAS");

    // gated_activation_kernels.h
    m.def("gated_activation", &ds_gated_activation, "DeepScale gated activation in CUDA");

    // rms_norm.h
    m.def("rms_norm", &rms_norm, "DeepScale rms norm in CUDA");
    m.def("rms_pre_norm", &rms_pre_norm, "DeepScale rms pre norm in CUDA");

    // linear_kernels.h
    m.def("cuda_wf6af16_linear", &cuda_wf6af16_linear, "DeepScale Wf6Af16 linear in CUDA");
    m.def(
        "preprocess_weight", &preprocess_weight, "preprocess the FP16 weight to be 2bit and 4 bit");
}
