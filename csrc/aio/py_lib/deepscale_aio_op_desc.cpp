// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

#include "deepscale_aio_op_desc.h"

using namespace std;

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const int64_t file_num_bytes,
                           const int intra_op_parallelism,
                           const bool validate)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename(filename),
      _file_num_bytes(file_num_bytes),
      _intra_op_parallelism(intra_op_parallelism),
      _num_bytes_per_thread(file_num_bytes / intra_op_parallelism),
      _validate(validate)
{
}

char* io_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void io_op_desc_t::finish() {}

void io_op_desc_t::validate() {}

void io_op_desc_t::run(const int tid,
                       std::unique_ptr<aio_context>& aio_ctxt,
                       deepscale_aio_config_t* aio_config)
{
}
