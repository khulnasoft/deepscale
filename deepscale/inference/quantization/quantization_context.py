# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from deepscale.runtime.zero import partition_parameters
from deepscale.runtime.swap_tensor.partitioned_param_swapper import AsyncPartitionedParameterSwapper


class QuantizationContext(partition_parameters.Init):

    def __init__(self, config_dict_or_path, param_swapper: AsyncPartitionedParameterSwapper = None) -> None:
        super().__init__(config_dict_or_path=config_dict_or_path, param_swapper=param_swapper)
