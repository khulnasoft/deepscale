# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import pytest
import torch
import deepscale
from unit.common import DistributedTest
from unit.simple_model import create_config_from_dict


@pytest.mark.inference
class TestInferenceConfig(DistributedTest):
    world_size = 1

    def test_overlap_kwargs(self):
        config = {"replace_with_kernel_inject": True, "dtype": torch.float32}
        kwargs = {"replace_with_kernel_inject": True}

        engine = deepscale.init_inference(torch.nn.Module(), config=config, **kwargs)
        assert engine._config.replace_with_kernel_inject

    def test_overlap_kwargs_conflict(self):
        config = {"replace_with_kernel_inject": True}
        kwargs = {"replace_with_kernel_inject": False}

        with pytest.raises(ValueError):
            engine = deepscale.init_inference(torch.nn.Module(), config=config, **kwargs)

    def test_kwargs_and_config(self):
        config = {"replace_with_kernel_inject": True}
        kwargs = {"dtype": torch.float32}

        engine = deepscale.init_inference(torch.nn.Module(), config=config, **kwargs)
        assert engine._config.replace_with_kernel_inject
        assert engine._config.dtype == kwargs["dtype"]

    def test_json_config(self, tmpdir):
        config = {"replace_with_kernel_inject": True, "dtype": "torch.float32"}
        config_json = create_config_from_dict(tmpdir, config)

        engine = deepscale.init_inference(torch.nn.Module(), config=config_json)
        assert engine._config.replace_with_kernel_inject
