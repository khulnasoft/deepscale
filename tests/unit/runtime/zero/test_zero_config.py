# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from deepscale.runtime.zero.config import DeepScaleZeroConfig, DeepScaleZeroOffloadParamConfig, DeepScaleZeroOffloadOptimizerConfig


def test_zero_config_deprecatedfields():
    config = DeepScaleZeroConfig(**{"cpu_offload_param": True})
    assert isinstance(config.offload_param, DeepScaleZeroOffloadParamConfig)

    config = DeepScaleZeroConfig(**{"cpu_offload": True})
    assert isinstance(config.offload_optimizer, DeepScaleZeroOffloadOptimizerConfig)

    config = DeepScaleZeroConfig(**{"stage3_gather_fp16_weights_on_model_save": True})
    assert config.gather_16bit_weights_on_model_save == True


def test_zero_config_aliasfields():
    config = DeepScaleZeroConfig(**{"stage3_prefetch_bucket_size": 12345})
    assert config.prefetch_bucket_size == 12345

    config = DeepScaleZeroConfig(**{"stage3_param_persistence_threshold": 12345})
    assert config.param_persistence_threshold == 12345

    config = DeepScaleZeroConfig(**{"stage3_max_reuse_distance": 12345})
    assert config.max_reuse_distance == 12345

    config = DeepScaleZeroConfig(**{"stage3_gather_16bit_weights_on_model_save": True})
    assert config.gather_16bit_weights_on_model_save == True


def test_zero_config_pipeline_loading_checkpoint():
    for stage in [0, 1, 2]:
        config = DeepScaleZeroConfig(**{"stage": stage})
        assert config.pipeline_loading_checkpoint == False


def test_zero_config_overlapcomm():
    for stage in [0, 1, 2]:
        config = DeepScaleZeroConfig(**{"stage": stage})
        assert config.overlap_comm == False

    config = DeepScaleZeroConfig(**{"stage": 3})
    assert config.overlap_comm == True


def test_zero_config_offload_configs():
    config = DeepScaleZeroConfig()
    assert config.offload_param is None
    assert config.offload_optimizer is None

    config = DeepScaleZeroConfig(**{"offload_param": None, "offload_optimizer": None})
    assert config.offload_param is None
    assert config.offload_optimizer is None

    config = DeepScaleZeroConfig(**{"offload_param": {}, "offload_optimizer": {}})
    assert isinstance(config.offload_param, DeepScaleZeroOffloadParamConfig)
    assert isinstance(config.offload_optimizer, DeepScaleZeroOffloadOptimizerConfig)


def test_zero_offload_optimizer_config_pipeline():
    config = DeepScaleZeroOffloadOptimizerConfig()
    assert config.pipeline == False

    config = DeepScaleZeroOffloadOptimizerConfig(**{"pipeline_read": True, "pipeline_write": False})
    assert config.pipeline == True

    config = DeepScaleZeroOffloadOptimizerConfig(**{"pipeline_read": False, "pipeline_write": True})
    assert config.pipeline == True

    config = DeepScaleZeroOffloadOptimizerConfig(**{"pipeline_read": True, "pipeline_write": True})
    assert config.pipeline == True
