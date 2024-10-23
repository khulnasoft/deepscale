# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team
from .builder import SYCLOpBuilder


class PackbitsBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_PACK_BITS"
    NAME = "pack_bits"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepscale.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/xpu/packbits/packing.cpp']

    def include_paths(self):
        return ['csrc/xpu/includes']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()
