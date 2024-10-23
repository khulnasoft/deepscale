# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from pydantic import Field
from typing import Optional

from deepscale.runtime.config_utils import DeepScaleConfigModel
from .ragged import DSStateManagerConfig


class DeepScaleTPConfig(DeepScaleConfigModel):
    """ Configure tensor parallelism settings """

    tp_size: int = 1
    """ Number of devices to split the model across using tensor parallelism. """


class QuantizationConfig(DeepScaleConfigModel):
    """ Configure tensor parallelism settings """

    quantization_mode: Optional[str] = None
    """ The quantization mode in string format. The supported modes are as follows:
        - 'wf6af16', weight-only quantization with FP6 weight and FP16 activation.
    """
    # TODO: may reuse the constants in deepscale/compression/constants.py


class RaggedInferenceEngineConfig(DeepScaleConfigModel):
    """ Sets parameters for DeepScale Inference Engine. """

    tensor_parallel: DeepScaleTPConfig = Field({}, alias="tp")
    """
    Configuration for tensor parallelism used to split the model across several
    GPUs. Expects a dictionary containing values for :any:`DeepScaleTPConfig`.
    """

    state_manager: DSStateManagerConfig = Field({}, alias="manager")
    """
    Configuration for managing persistent state
    """

    quantization: QuantizationConfig = {}
