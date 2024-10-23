# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import torch.nn as nn


class DeepScaleTransformerBase(nn.module):

    def __init__(self):
        pass

    # this would be the new clean base class that will replace DeepScaleTransformerInference.
    # we currently don't know how this will look like but keeping it here as a placeholder.
