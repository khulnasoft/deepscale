// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepscale_py_aio_handle.h"
#include <cstdlib>

using namespace std;

deepscale_aio_handle_t::deepscale_aio_handle_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const int intra_op_parallelism)
    : deepscale_io_handle_t(block_size,
                            queue_depth,
                            single_submit,
                            overlap_events,
                            intra_op_parallelism)
{
}

deepscale_aio_handle_t::~deepscale_aio_handle_t() {}
