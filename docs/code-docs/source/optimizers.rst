Optimizers
===================

DeepScale offers high-performance implementations of ``Adam`` optimizer on CPU; ``FusedAdam``, ``FusedLamb``, ``OnebitAdam``, ``OnebitLamb`` optimizers on GPU.

Adam (CPU)
----------------------------
.. autoclass:: deepscale.ops.adam.DeepScaleCPUAdam

FusedAdam (GPU)
----------------------------
.. autoclass:: deepscale.ops.adam.FusedAdam

FusedLamb (GPU)
----------------------------
.. autoclass:: deepscale.ops.lamb.FusedLamb

OneBitAdam (GPU)
----------------------------
.. autoclass:: deepscale.runtime.fp16.onebit.adam.OnebitAdam

ZeroOneAdam (GPU)
----------------------------
.. autoclass:: deepscale.runtime.fp16.onebit.zoadam.ZeroOneAdam

OnebitLamb (GPU)
----------------------------
.. autoclass:: deepscale.runtime.fp16.onebit.lamb.OnebitLamb
