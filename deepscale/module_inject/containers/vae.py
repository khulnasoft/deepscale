# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from ..policy import DSPolicy
from ...model_implementations.diffusers.vae import DSVAE


class VAEPolicy(DSPolicy):

    def __init__(self):
        super().__init__()
        try:
            import diffusers
            if hasattr(diffusers.models, "autoencoders"):
                # Diffusers >= 0.25.0
                # Changes location to 'autoencoders' directory
                self._orig_layer_class = diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL
            elif hasattr(diffusers.models.vae, "AutoencoderKL"):
                # Diffusers < 0.12.0
                self._orig_layer_class = diffusers.models.vae.AutoencoderKL
            else:
                # Diffusers >= 0.12.0 & < 0.25.0
                # Changes location of AutoencoderKL
                self._orig_layer_class = diffusers.models.autoencoder_kl.AutoencoderKL
        except ImportError:
            self._orig_layer_class = None

    def match(self, module):
        return isinstance(module, self._orig_layer_class)

    def match_replaced(self, module):
        return isinstance(module, DSVAE)

    def apply(self, module, enable_cuda_graph=True):
        # TODO(cmikeh2): Enable cuda graph should be an inference configuration
        return DSVAE(module, enable_cuda_graph=enable_cuda_graph)

    # NOTE (lekurile): Should we have a diffusers policy class?
    def attention(self, client_module):
        pass
