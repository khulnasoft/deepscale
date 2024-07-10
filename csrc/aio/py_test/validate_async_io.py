"""
Copyright 2024 The KhulnaSoft DeepScale Team
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import deepscale
from deepscale.ops.aio import AsyncIOBuilder

assert AsyncIOBuilder().is_compatible()
