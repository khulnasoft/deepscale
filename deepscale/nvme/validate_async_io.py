# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""
from deepscale.ops.op_builder import AsyncIOBuilder
assert AsyncIOBuilder().is_compatible()
assert AsyncIOBuilder().load()
