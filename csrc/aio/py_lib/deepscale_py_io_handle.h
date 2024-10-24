// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <condition_variable>
#include <memory>
#include "deepscale_aio_thread.h"
#include "deepscale_pin_tensor.h"

struct deepscale_io_handle_t {
    std::unique_ptr<struct aio_context> _aio_ctxt;
    const bool _single_submit;
    const bool _overlap_events;
    const int _intra_op_parallelism;
    deepscale_aio_config_t _aio_config;

    std::vector<std::shared_ptr<struct deepscale_aio_thread_t>> _thread_contexts;
    std::vector<std::thread> _threads;
    int _num_pending_ops;
    std::unique_ptr<struct deepscale_pin_tensor_t> _pinned_tensor_mgr;

    deepscale_io_handle_t(const int block_size,
                          const int queue_depth,
                          const bool single_submit,
                          const bool overlap_events,
                          const int intra_op_parallelism);

    virtual ~deepscale_io_handle_t() = 0;

    const int get_block_size() const;
    const int get_queue_depth() const;
    const bool get_single_submit() const;
    const bool get_overlap_events() const;
    const int get_intra_op_parallelism() const;

    int read(torch::Tensor& buffer, const char* filename, const bool validate);

    int write(const torch::Tensor& buffer, const char* filename, const bool validate);

    int pread(const torch::Tensor& buffer,
              const char* filename,
              const bool validate,
              const bool async);

    int pwrite(const torch::Tensor& buffer,
               const char* filename,
               const bool validate,
               const bool async);

    int sync_pread(torch::Tensor& buffer, const char* filename);

    int sync_pwrite(const torch::Tensor& buffer, const char* filename);

    int async_pread(torch::Tensor& buffer, const char* filename);

    int async_pwrite(const torch::Tensor& buffer, const char* filename);

    // TODO: Make API's args to be shape and dtype.
    torch::Tensor new_cpu_locked_tensor(const int64_t num_elem,
                                        const torch::Tensor& example_tensor);

    bool free_cpu_locked_tensor(torch::Tensor&);

    int wait();

    void _stop_threads();

    void _schedule_aio_work(std::shared_ptr<struct io_op_desc_t> scheduled_op);

    std::shared_ptr<struct io_op_desc_t> _wait_for_aio_work();

    bool _is_valid_parallel_aio_op(const bool read_op, const int64_t num_bytes);

    virtual std::shared_ptr<struct io_op_desc_t> _create_io_op_desc(const bool read_op,
                                                                    const torch::Tensor& buffer,
                                                                    const int fd,
                                                                    const char* filename,
                                                                    const int64_t file_num_bytes,
                                                                    const bool validate);
};
