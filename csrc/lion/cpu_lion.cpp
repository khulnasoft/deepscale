// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

#include "cpu_lion.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lion_update", &ds_lion_step, "DeepScale CPU Lion update (C++)");
    m.def("create_lion", &create_lion_optimizer, "DeepScale CPU Lion (C++)");
    m.def("destroy_lion", &destroy_lion_optimizer, "DeepScale CPU Lion destroy (C++)");
}
