// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

#include <memory>
#include <queue>
#include "deepscale_aio_op_desc.h"

struct cpu_op_desc_t : io_op_desc_t {
    torch::Tensor _cpu_buffer;
    bool _use_bounce_buffer;
    bool _is_managed_bounce_buffer;
    const std::unique_ptr<struct deepscale_pin_tensor_t>& _pinned_tensor_mgr;

    cpu_op_desc_t(const bool read_op,
                  const torch::Tensor& buffer,
                  const std::unique_ptr<struct deepscale_pin_tensor_t>& pinned_tensor_mgr,
                  const int fd,
                  const char* filename,
                  const int64_t file_num_bytes,
                  const int intra_op_parallelism,
                  const bool validate);

    void run(const int tid,
             std::unique_ptr<aio_context>& aio_ctxt,
             deepscale_aio_config_t* aio_config);

    char* data_ptr() const;

    void validate();

    void finish();

    void _alloc_bounce_buffer();
    void _free_bounce_buffer();
};
