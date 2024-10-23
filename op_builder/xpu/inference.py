# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team
from .builder import SYCLOpBuilder


class InferenceBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepscale.ops.transformer.inference.{self.NAME}_op'

    def sources(self):
        return

    def include_paths(self):
        return []

    def extra_ldflags(self):
        return []

    def cxx_args(self):
        return []

    def load(self):
        try:
            import intel_extension_for_pytorch
            if hasattr(intel_extension_for_pytorch, "deepscale"):
                return intel_extension_for_pytorch.deepscale.transformer_inference.transformer_inference
            else:
                return intel_extension_for_pytorch.xpu.deepscale
        except ImportError:
            raise ImportError("Please install intel-extension-for-pytorch >= 2.1.30 to include DeepScale kernels.")
