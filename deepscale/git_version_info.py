# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

try:
    #  This is populated by setup.py
    from .git_version_info_installed import *  # noqa: F401 # type: ignore
except ModuleNotFoundError:
    import os
    if os.path.isfile('version.txt'):
        # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
        version = open('version.txt', 'r').read().strip()
    else:
        version = "0.0.0"
    git_hash = '[none]'
    git_branch = '[none]'

    from .ops.op_builder.all_ops import ALL_OPS
    installed_ops = dict.fromkeys(ALL_OPS.keys(), False)
    accelerator_name = ""
    torch_info = {'version': "0.0", "cuda_version": "0.0", "hip_version": "0.0"}

# compatible_ops list is recreated for each launch
from .ops.op_builder.all_ops import ALL_OPS

compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()
    compatible_ops[op_name] = op_compatible
    compatible_ops["deepscale_not_implemented"] = False
