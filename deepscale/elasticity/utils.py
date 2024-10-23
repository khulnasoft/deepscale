# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from deepscale.utils.torch import required_torch_version


def is_torch_elastic_compatible():
    '''
        Helper to lookup torch version. Elastic training is
        introduced in 1.11.x
    '''
    return required_torch_version(min_version=1.11)
