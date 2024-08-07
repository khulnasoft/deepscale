"""
Copyright 2024 The KhulnaSoft DeepScale Team
"""

from .builder import CUDAOpBuilder


class FusedLambBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_LAMB"
    NAME = "fused_lamb"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"deepscale.ops.lamb.{self.NAME}_op"

    def sources(self):
        return ["csrc/lamb/fused_lamb_cuda.cpp", "csrc/lamb/fused_lamb_cuda_kernel.cu"]

    def include_paths(self):
        return ["csrc/includes"]

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        return (["-lineinfo",
                 "-O3",
                 "--use_fast_math"] + self.version_dependent_macros() +
                self.compute_capability_args())
