/*
Copyright 2024 The KhulnaSoft DeepScale Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <torch/extension.h>
#include "deepscale_py_aio_handle.h"
#include "deepscale_py_copy.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("aio_read", &deepscale_py_aio_read, "DeepScale Asynchornous I/O Read");

    m.def("aio_write", &deepscale_py_aio_write, "DeepScale Asynchornous I/O Write");

    m.def("deepscale_memcpy", &deepscale_py_memcpy, "DeepScale Memory Copy");

    py::class_<deepscale_aio_handle_t>(m, "aio_handle")
        .def(py::init<const int, const int, const bool, const bool, const int>())

        .def("get_block_size", &deepscale_aio_handle_t::get_block_size)
        .def("get_queue_depth", &deepscale_aio_handle_t::get_queue_depth)
        .def("get_single_submit", &deepscale_aio_handle_t::get_single_submit)
        .def("get_overlap_events", &deepscale_aio_handle_t::get_overlap_events)
        .def("get_thread_count", &deepscale_aio_handle_t::get_thread_count)

        .def("read", &deepscale_aio_handle_t::read)
        .def("write", &deepscale_aio_handle_t::write)

        .def("pread", &deepscale_aio_handle_t::pread)
        .def("pwrite", &deepscale_aio_handle_t::pwrite)

        .def("sync_pread", &deepscale_aio_handle_t::sync_pread)
        .def("sync_pwrite", &deepscale_aio_handle_t::sync_pwrite)
        .def("async_pread", &deepscale_aio_handle_t::async_pread)
        .def("async_pwrite", &deepscale_aio_handle_t::async_pwrite)

        .def("wait", &deepscale_aio_handle_t::wait);
}
