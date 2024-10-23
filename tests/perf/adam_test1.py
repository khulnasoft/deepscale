# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import torch
from deepscale.ops.adam import DeepScaleCPUAdam
import time

device = 'cpu'
model_size = 1 * 1024**3
param = torch.nn.Parameter(torch.ones(model_size, device=device))

optimizer = DeepScaleCPUAdam([param])
#torch.set_num_threads(128)
param.grad = torch.ones(model_size, device=device)
avg = 0
for i in range(100):
    start = time.time()
    optimizer.step()
    stop = time.time()
    avg += (stop - start)
    param.grad = torch.ones(model_size, device=device) * 2
print("Elapsed Time is ", avg / 100)
