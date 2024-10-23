// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "DeepScale CPU Adam update (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepScale CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepScale CPU Adam destroy (C++)");
}
