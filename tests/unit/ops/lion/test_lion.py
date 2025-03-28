# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import deepscale
import torch
import pytest

from deepscale.ops.lion import FusedLion
from deepscale.ops.lion import DeepScaleCPULion
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepscale.accelerator import get_accelerator
from deepscale.ops.op_builder import CPULionBuilder

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)
# yapf: disable
#'optimizer, zero_offload, resulting_optimizer
lion_configs = [["Lion",  False, FusedLion],
                ["Lion",  True,  DeepScaleCPULion]]

@pytest.mark.parametrize(
    'optimizer, zero_offload, resulting_optimizer',
    lion_configs)
class TestLionConfigs(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    @pytest.mark.skipif(not deepscale.ops.__compatible_ops__[CPULionBuilder.NAME], reason="CPULionBuilder has not been implemented on this system.")
    def test(self,
             optimizer,
             zero_offload,
             resulting_optimizer):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": optimizer,
                "params": {
                    "lr": 0.00015,
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "cpu_offload": zero_offload
            }
        }
        model = SimpleModel(10)
        model, _, _, _ = deepscale.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        # get base optimizer under zero
        ds_optimizer = model.optimizer.optimizer
        opt_class = resulting_optimizer
        assert isinstance(ds_optimizer, opt_class)
