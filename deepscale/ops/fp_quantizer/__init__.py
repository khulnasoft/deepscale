# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from .quantize import FP_Quantize, Quantizer

try:
    import triton
    from .fp8_gemm import matmul_fp8
except ImportError:
    pass
