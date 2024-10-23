// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

/*
Functionality for swapping tensors to/from (NVMe) storage devices.
*/

#include <deepscale_aio_common.h>
#include <stdlib.h>
#include <torch/extension.h>

int deepscale_py_aio_write(const torch::Tensor& buffer,
                           const char* filename,
                           const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const bool validate);

int deepscale_py_aio_read(torch::Tensor& buffer,
                          const char* filename,
                          const int block_size,
                          const int queue_depth,
                          const bool single_submit,
                          const bool overlap_events,
                          const bool validate);
