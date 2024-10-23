# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import torch
from ..config import DeepScaleInferenceConfig
from .base import BaseOp


class EinsumSecSmEcmOp(BaseOp):

    def __init__(self, config: DeepScaleInferenceConfig):
        super(EinsumSecSmEcmOp, self).__init__(config)

        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.einsum_sec_sm_ecm_func = self.inference_module.einsum_sec_sm_ecm_fp16
            else:
                self.einsum_sec_sm_ecm_func = self.inference_module.einsum_sec_sm_ecm_fp32
        except AttributeError:
            self.einsum_sec_sm_ecm_func = self.einsum_sec_sm_ecm_fallback

    @classmethod
    def einsum_sec_sm_ecm_fallback(cls, Q, W):
        raise NotImplementedError("einsum sec sm ecm fallback isn't implemented")

    def forward(self, Q, W):
        return self.einsum_sec_sm_ecm_func(Q, W)
