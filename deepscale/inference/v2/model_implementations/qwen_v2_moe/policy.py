# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from typing import Any

from ...config_v2 import RaggedInferenceEngineConfig
from ..inference_policy_base import ContainerMap, InferenceV2Policy
from .container import Qwen2MoeNonTransformerContainer, Qwen2MoeTransformerContainer
from .model import Qwen2MoeInferenceModel


class Qwen2MoePolicy(InferenceV2Policy):

    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> Qwen2MoeInferenceModel:
        return Qwen2MoeInferenceModel(config=self._model_config, engine_config=engine_config, base_mp_group=mp_group)

    def build_container_map(self) -> ContainerMap:
        map = ContainerMap()

        transformer_containers = [Qwen2MoeTransformerContainer(self.model) for _ in range(self.model.num_layers)]

        map.set_transformer_params(['model.layers'], transformer_containers)

        map.set_non_transformer_params(Qwen2MoeNonTransformerContainer(self.model))

        map.set_unmapped_params([])

        return map
