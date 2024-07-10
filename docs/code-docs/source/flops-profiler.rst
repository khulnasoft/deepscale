Flops Profiler

==============

The flops profiler in DeepScale profiles the forward pass of a model and measures its parameters, latency, and floating point operations. The DeepScale flops profiler can be used with the DeepScale runtime or as a standalone package.

When using DeepScale for model training, the flops profiler can be configured in the deepscale_config file without user code changes. To use the flops profiler outside of the DeepScale runtime, one can simply install DeepScale and import the flops_profiler package to use the APIs directly.

Please see the `Flops Profiler tutorial <https://www.deepscale.khulnasoft.com/tutorials/flops-profiler/>`_ for usage details.

Flops Profiler
---------------------------------------------------

.. automodule:: deepscale.profiling.flops_profiler.profiler
   :members:
   :show-inheritance:
