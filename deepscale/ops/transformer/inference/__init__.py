# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from .config import DeepScaleInferenceConfig
from ....model_implementations.transformers.ds_transformer import DeepScaleTransformerInference
from .moe_inference import DeepScaleMoEInferenceConfig, DeepScaleMoEInference
