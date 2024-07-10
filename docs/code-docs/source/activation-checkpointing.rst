Activation Checkpointing
========================

The activation checkpointing API's in DeepScale can be used to enable a range
of memory optimizations relating to activation checkpointing. These include
activation partitioning across GPUs when using model parallelism, CPU
checkpointing, contiguous memory optimizations, etc.

Please see the `DeepScale JSON config <https://www.deepscale.khulnasoft.com/docs/config-json/>`_
for the full set.

Here we present the activation checkpointing API. Please see the enabling
DeepScale for `Megatron-LM tutorial <https://www.deepscale.khulnasoft.com/tutorials/megatron/>`_
for example usage.

Configuring Activation Checkpointing
------------------------------------
.. autofunction:: deepscale.checkpointing.configure

.. autofunction:: deepscale.checkpointing.is_configured


Using Activation Checkpointing
------------------------------
.. autofunction:: deepscale.checkpointing.checkpoint

.. autofunction:: deepscale.checkpointing.reset


Configuring and Checkpointing Random Seeds
------------------------------------------
.. autofunction:: deepscale.checkpointing.get_cuda_rng_tracker

.. autofunction:: deepscale.checkpointing.model_parallel_cuda_manual_seed

.. autoclass:: deepscale.checkpointing.CudaRNGStatesTracker

.. autoclass:: deepscale.checkpointing.CheckpointFunction
