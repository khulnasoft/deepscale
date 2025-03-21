// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

#include <cstdlib>
#include <fstream>
#include <memory>
#include <queue>
#include <set>
#include <string>

#include "deepscale_aio_op_desc.h"
#include "deepscale_gds_utils.h"

struct gds_op_desc_t : io_op_desc_t {
    CUfileDescr_t _cf_descr;
    CUfileHandle_t _cf_handle;
    void* _base_ptr;

    gds_op_desc_t(const bool read_op,
                  const torch::Tensor& buffer,
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

    void _report_error(const ssize_t return_code, const int error_num, const off_t offset);

    static void add_buffer_to_registry(const torch::Tensor& buffer);

    static void remove_buffer_from_registry(const torch::Tensor& buffer);
};
