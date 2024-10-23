// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepScale Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <torch/extension.h>
#include "deepscale_py_aio_handle.h"
#include "deepscale_py_copy.h"
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("aio_read", &deepscale_py_aio_read, "DeepScale Asynchronous I/O Read");

    m.def("aio_write", &deepscale_py_aio_write, "DeepScale Asynchronous I/O Write");

    m.def("deepscale_memcpy", &deepscale_py_memcpy, "DeepScale Memory Copy");

    py::class_<deepscale_aio_handle_t>(m, "aio_handle")
        .def(py::init<const int, const int, const bool, const bool, const int>(),
             "AIO handle constructor",
             "block_size"_a = 1024 * 1024,
             "queue_depth"_a = 128,
             "single_submit"_a = false,
             "overlap_events"_a = false,
             "intra_op_parallelism"_a = 1)

        .def("get_block_size", &deepscale_aio_handle_t::get_block_size)
        .def("get_queue_depth", &deepscale_aio_handle_t::get_queue_depth)
        .def("get_single_submit", &deepscale_aio_handle_t::get_single_submit)
        .def("get_overlap_events", &deepscale_aio_handle_t::get_overlap_events)
        .def("get_intra_op_parallelism", &deepscale_aio_handle_t::get_intra_op_parallelism)

        .def("read",
             &deepscale_aio_handle_t::read,
             "Synchronous and non-parallel file read. Returns count of completed read ops",
             "buffer"_a,
             "filename"_a,
             "validate"_a)

        .def("write",
             &deepscale_aio_handle_t::write,
             "Synchronous and non-parallel file write. Returns count of completed write ops",
             "buffer"_a,
             "filename"_a,
             "validate"_a)

        .def("pread",
             &deepscale_aio_handle_t::pread,
             "Parallel file read with option of parallelism. Returns count of completed read ops",
             "buffer"_a,
             "filename"_a,
             "validate"_a,
             "async"_a)

        .def("pwrite",
             &deepscale_aio_handle_t::pwrite,
             "Parallel file write with option of parallelism. Returns count of completed write ops",
             "buffer"_a,
             "filename"_a,
             "validate"_a,
             "async"_a)

        .def("sync_pread",
             &deepscale_aio_handle_t::sync_pread,
             "Synchrononous parallel file read. Returns count of completed read ops",
             "buffer"_a,
             "filename"_a)

        .def("sync_pwrite",
             &deepscale_aio_handle_t::sync_pwrite,
             "Synchronous parallel file write. Returns count of completed write ops",
             "buffer"_a,
             "filename"_a)

        .def("async_pread",
             &deepscale_aio_handle_t::async_pread,
             "Asynchronous parallel file read. Returns 0 on success. Returns 0 on success, and "
             "following wait() returns count of completed ops.",
             "buffer"_a,
             "filename"_a)

        .def("async_pwrite",
             &deepscale_aio_handle_t::async_pwrite,
             "Asynchronous parallel file write. Returns 0 on success, and following wait() returns "
             "count of completed ops.",
             "buffer"_a,
             "filename"_a)

        .def("new_cpu_locked_tensor",
             &deepscale_aio_handle_t::new_cpu_locked_tensor,
             "Allocate pinned CPU tensor.",
             "num_elem"_a,
             "example_tenosr"_a)

        .def("free_cpu_locked_tensor",
             &deepscale_aio_handle_t::free_cpu_locked_tensor,
             "Free pinned CPU tensor.",
             "tensor"_a)

        .def("wait",
             &deepscale_aio_handle_t::wait,
             "Wait for (ongoing) asynchronous operations to complete");
}
