# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from deepscale.utils import log_dist


# helper function to map between DS policies and DS containers
def policy_to_ds_container(**kwargs):
    from .containers import HFGPT2LayerPolicy, DS_GPT2Container
    from .containers import HFBertLayerPolicy, DS_BERTContainer
    from .containers import BLOOMLayerPolicy, DS_BloomContainer
    from .containers import HFGPTJLayerPolicy, DS_GPTJContainer
    from .containers import HFGPTNEOLayerPolicy, DS_GPTNEOContainer
    from .containers import GPTNEOXLayerPolicy, DS_GPTNEOXContainer
    from .containers import HFOPTLayerPolicy, DS_OPTContainer
    from .containers import MegatronLayerPolicy, DS_MegatronGPTContainer
    from .containers import HFDistilBertLayerPolicy, DS_DistilBERTContainer
    from .containers import LLAMALayerPolicy, DS_LLAMAContainer
    from .containers import LLAMA2LayerPolicy, DS_LLAMA2Container
    from .containers import InternLMLayerPolicy, DS_InternLMContainer

    policy_to_container = {
        HFGPT2LayerPolicy: DS_GPT2Container,
        HFBertLayerPolicy: DS_BERTContainer,
        BLOOMLayerPolicy: DS_BloomContainer,
        HFGPTJLayerPolicy: DS_GPTJContainer,
        HFGPTNEOLayerPolicy: DS_GPTNEOContainer,
        GPTNEOXLayerPolicy: DS_GPTNEOXContainer,
        HFOPTLayerPolicy: DS_OPTContainer,
        MegatronLayerPolicy: DS_MegatronGPTContainer,
        HFDistilBertLayerPolicy: DS_DistilBERTContainer,
        LLAMALayerPolicy: DS_LLAMAContainer,
        LLAMA2LayerPolicy: DS_LLAMA2Container,
        InternLMLayerPolicy: DS_InternLMContainer
    }

    container = None
    policy = kwargs['policy']
    assert policy is not None, "Policy cannot be None"
    policy_type = type(policy)

    if policy_type not in policy_to_container:
        log_dist(f"Policy type {policy_type} not supported", [0])
    else:
        container = policy_to_container[policy_type](**kwargs)

    return container
