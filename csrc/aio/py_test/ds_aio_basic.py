# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import torch
import os
import time
from deepscale.ops.aio import AsyncIOBuilder
from multiprocessing import Pool, Barrier
from test_ds_aio_utils import report_results, task_log, task_barrier


def pre_basic(args, tid, read_op):
    io_string = "Read" if read_op else "Write"
    num_bytes = os.path.getsize(args.read_file) if read_op else args.write_size
    file = args.read_file if read_op else f'{args.write_file}.{tid}'

    task_log(tid, f'Allocate tensor of size {num_bytes} bytes')
    buffer = torch.empty(num_bytes, dtype=torch.uint8, device='cpu').pin_memory()
    task_log(tid, f'{io_string} file {file} of size {num_bytes} bytes from buffer on device {buffer.device}')

    ctxt = {}
    ctxt['file'] = file
    ctxt['num_bytes'] = num_bytes
    ctxt['buffer'] = buffer
    ctxt['elapsed_sec'] = 0

    return ctxt


def pre_basic_read(pool_params):
    args, tid = pool_params
    ctxt = pre_basic(args, tid, True)
    return ctxt


def pre_basic_write(pool_params):
    args, tid = pool_params
    ctxt = pre_basic(args, tid, False)
    return ctxt


def post_basic(pool_params):
    _, _, ctxt = pool_params
    ctxt["buffer"].detach()
    ctxt["buffer"] = None
    return ctxt


def main_basic_read(pool_params):
    args, tid, ctxt = pool_params
    start_time = time.time()
    AsyncIOBuilder().load().aio_read(ctxt['buffer'], ctxt['file'], args.block_size, args.queue_depth,
                                     args.single_submit, not args.sequential_requests, args.validate)
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def main_basic_write(pool_params):
    args, tid, ctxt = pool_params
    start_time = time.time()
    AsyncIOBuilder().load().aio_write(ctxt['buffer'], ctxt['file'], args.block_size, args.queue_depth,
                                      args.single_submit, not args.sequential_requests, args.validate)
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def get_schedule(args, read_op):
    schedule = {}
    if read_op:
        schedule['pre'] = pre_basic_read
        schedule['post'] = post_basic
        schedule['main'] = main_basic_read
    else:
        schedule['pre'] = pre_basic_write
        schedule['post'] = post_basic
        schedule['main'] = main_basic_write

    return schedule


def _aio_handle_tasklet(pool_params):
    args, tid, read_op = pool_params
    num_processes = len(args.mapping_dict)

    # Create schedule
    schedule = get_schedule(args, read_op)
    task_log(tid, f'schedule = {schedule}')
    task_barrier(aio_barrier, num_processes)

    # Run pre task
    task_log(tid, f'running pre-task')
    ctxt = schedule["pre"]((args, tid))
    task_barrier(aio_barrier, num_processes)

    # Run main tasks in a loop
    ctxt["main_task_sec"] = 0
    for i in range(args.loops):
        task_log(tid, f'running main task {i}')
        start_time = time.time()
        ctxt = schedule["main"]((args, tid, ctxt))
        task_barrier(aio_barrier, num_processes)
        stop_time = time.time()
        ctxt["main_task_sec"] += stop_time - start_time

    # Run post task
    task_log(tid, f'running post-task')
    ctxt = schedule["post"]((args, tid, ctxt))
    task_barrier(aio_barrier, num_processes)

    return ctxt["main_task_sec"], ctxt["elapsed_sec"], ctxt["num_bytes"] * args.loops


def _init_tasklet(b):
    global aio_barrier
    aio_barrier = b


def aio_basic_multiprocessing(args, read_op):
    num_processes = len(args.mapping_dict)
    b = Barrier(num_processes)
    pool_params = [(args, p, read_op) for p in range(num_processes)]
    with Pool(processes=num_processes, initializer=_init_tasklet, initargs=(b, )) as p:
        pool_results = p.map(_aio_handle_tasklet, pool_params)

    report_results(args, read_op, pool_results)
