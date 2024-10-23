# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import torch

from transformers import AutoModelForCausalLM
import deepscale
import argparse
from deepscale.accelerator import get_accelerator

deepscale.runtime.utils.see_memory_usage('pre test', force=True)

model = AutoModelForCausalLM.from_pretrained('facebook/opt-350M').half().to(get_accelerator().device_name())
parser = argparse.ArgumentParser()
parser = deepscale.add_config_arguments(parser)
args = parser.parse_args()

deepscale.runtime.utils.see_memory_usage('post test', force=True)

m, _, _, _ = deepscale.initialize(model=model, args=args, enable_hybrid_engine=True)

m.eval()
input = torch.ones(1, 16, device='cuda', dtype=torch.long)
out = m(input)

m.train()
out = m(input)
print(out['logits'], out['logits'].norm())
