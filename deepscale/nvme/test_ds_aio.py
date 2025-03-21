# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import multiprocessing as mp
from .ds_aio_basic import aio_basic_multiprocessing
from .ds_aio_handle import aio_handle_multiprocessing
from .ds_aio_args import get_validated_args


def ds_io_main():
    print(f'Testing deepscale_aio python frontend')

    args = get_validated_args()
    mp.set_start_method('spawn')
    multiprocess_function = aio_handle_multiprocessing if args.handle else aio_basic_multiprocessing
    multiprocess_function(args, args.read)


if __name__ == "__main__":
    ds_io_main()
