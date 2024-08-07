ZeRO
####

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across
data-parallel processes by partitioning the three model states (optimizer
states, gradients, and parameters) across data-parallel processes instead of
replicating them. By doing this, it boosts memory efficiency compared to
classic data-parallelism while retaining its computational granularity and
communication efficiency.

#. **ZeRO Stage 1**: The optimizer states (e.g., for `Adam optimizer <https://arxiv.org/abs/1412.6980>`_, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.

#. **ZeRO Stage 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.

#. **ZeRO Stage 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

In addition, ZeRO-3 includes the *infinity offload engine* to form
ZeRO-Infinity ([paper](https://arxiv.org/abs/2104.07857)), which can offload
all model states to both CPU and NVMe memory for huge memory savings.


For a deep dive of our algorithms, please see our `papers <https://www.deepscale.khulnasoft.com/#publications>`_ on `ZeRO
<https://arxiv.org/abs/1910.02054>`_, `ZeRO-Offload
<https://arxiv.org/abs/2101.06840>`_,
and `ZeRO-Infinity <https://arxiv.org/abs/2104.07857>`_.

.. note::
    DeepScale first included offloading capabilities with **ZeRO-Offload**, a
    system for offloading optimizer and gradient states to CPU memory within
    ZeRO-2. **ZeRO-Infinity** is the next generation of offloading
    capabilities, accessible to ZeRO-3. ZeRO-Infinity has all of the savings
    of ZeRO-Offload, plus is able to offload more the model weights and has
    more effective bandwidth utilization and overlapping of computation and
    communication.



Getting Started
---------------

If you are new to DeepScale, check out our `Getting Started <https://www.deepscale.khulnasoft.com/getting-started/>`_ page.

Once you are training with DeepScale, enabling ZeRO-3 offload is as simple as enabling it
in your DeepScale configuration! Below are a few examples of ZeRO-3 configurations. Please see
our `config guide <https://www.deepscale.khulnasoft.com/docs/config-json/#zero-optimizations-for-fp16-training>`_
for a complete list of options for configuration and performance tuning.

.. note::
        ZeRO-Infinity and ZeRO-Offload work best with our heavily optimized
        :class:`deepscale.ops.adam.DeepScaleCPUAdam` optimizer. We recommend using
        our `optimizer config <https://www.deepscale.khulnasoft.com/docs/config-json/#optimizer-parameters>`_
        to instruct :meth:`deepscale.initialize` to build the optimizer for you.



Example ZeRO-3 Configurations
=============================

#. Use ZeRO to partition the optimizer states (stage 1), gradients (stage 2),
   and parameters (stage 3).

    .. code-block:: python
        :emphasize-lines: 3

        {
            "zero_optimization": {
                "stage": 3,
            },
            "fp16": {
                "enabled": true
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                "lr": 0.001,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7
                }
            },
            ...
        }


#. Additionally offload the optimizer states and computations to the CPU with ZeRO-Infinity.

    .. code-block:: python

        {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu"
                }
            },
            ...
        }


#. Save even more memory by offloading parameters to the CPU memory.

    .. code-block:: python

        {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu"
                }
                "offload_param": {
                    "device": "cpu"
                }
            },
            ...
        }


#. Save even MORE memory by offloading to NVMe (if available on your system):

    .. code-block:: python

        {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "nvme",
                    "nvme_path": "/nvme_data"
                }
                "offload_param": {
                    "device": "nvme",
                    "nvme_path": "/nvme_data"
                }
            },
            ...
        }



Assumptions
===========

DeepScale automatically coordinates the collection (*i.e.,* all-gather),
partitioning (*i.e.,* scatter), and offloading of parameters at the
granularity of (sub)module ``forward()`` methods. The backward pass is
handled similarly. This strategy has two underlying assumptions:

#. The forward and backward passes of submodules must individually fit in device memory.
   If this not the case, :class:`deepscale.zero.TiledLinear` implements
   **memory-centric tiling** and works with ZeRO-3 to break linear layers
   into a sequence of smaller submodules that can fit in memory.

#. A module's parameters are only accessed within its own ``__init__`` and ``forward()`` methods.
   Otherwise, DeepScale must be instructed to collect and re-partition the parameter.
   See :ref:`external-parameters` for manually coordinating parameters.


Constructing Massive Models
---------------------------

ZeRO-3 enables massive models whose parameters exceed the size of individual
nodes in a system. For the typical case of training without model parallelism,
you can simply allocate your model in our context:

.. code-block:: python

    with deepscale.zero.Init():
        model = MyLargeModel()


.. autoclass:: deepscale.zero.Init
    :members:


.. _external-parameters:

Manual Parameter Coordination
-----------------------------

Most models require no modification to be trained with ZeRO-3. However, in
some cases one may need to access model weights outside of the training loop,
or to share weights across submodules during training. DeepScale has
several mechanisms to coordinate partitioned weights for ZeRO-3.


Gathering Parameters
====================

DeepScale provides mechanisms for collecting (or *gathering*) a partitioned parameter.

Some models partitioned with :class:`deepscale.zero.Init` may need to access
a module’s weights outside of the class constructor or its ``forward()``
method. We refer to these weights as **external parameters**, since these
parameters are accessed outside of the module that created them. To do so, use
:class:`deepscale.zero.GatheredParameters` or :meth:`deepscale.zero.register_external_parameter`.

.. autoclass:: deepscale.zero.GatheredParameters
    :members:


Registering External Parameters
===============================

ZeRO-3 will automatically collect and partition the model parameters as they
are needed during the forward and backward passes. However, in some cases a
parameter may be used outside of its module's forward pass. We call these
*external* parameters. ZeRO-3 can coordinate these parameters if they are
registered either automatically or manually.


.. note::
    DeepScale version ``0.3.15`` includes automatic external parameter
    discovery and registration to support the most common cases. Parameters
    can still be manually registered if they cannot be automatically
    detected.


DeepScale can automatically detect the following external parameter scenarios:


#. Parameter access: consider the following pattern common in language models such as GPT:

   The tensor ``embeddings.weight`` is used in both ``embeddings.forward()`` and
   ``compute_logits()``. We call ``embeddings.weight`` an *external* parameter
   because it is used in the training loop outside of its owning module's
   forward pass.


   .. code-block:: python

       class LanguageModel(torch.nn.Module):
           ...
           def forward(self, inputs):
               embeds = self.embeddings(inputs)
               ...
               logits = compute_logits(output, self.embeddings.weight)
               ...


#. Returning a parameter:

   ``CustomLinear`` returns both an output and its own ``bias`` parameter. DeepScale
   will detect the external ``bias`` parameter and register it with submodules that
   use ``CustomLinear``.

   .. code-block:: python

       class CustomLinear(torch.nn.Linear):
           def forward(self, *input):
               output = super().forward(*input)
               return output, self.bias



.. autofunction:: deepscale.zero.register_external_parameter

.. autofunction:: deepscale.zero.unregister_external_parameter


Memory-Centric Tiling
---------------------

To reduce the working memory requirements of DL training for large models,
ZeRO-Infinity includes technique called *memory-centric tiling* that exploits
the data fetch and release pattern of ZeRO-3 to reduce the working memory
requirements by breaking down a large operator into smaller tiles that can be
executed sequentially. When combined with ZeRO-3, the parameter and gradients
of each tile can be fetched and released one at a time, reducing the working
memory proportional to the number of tiles. Therefore, ZeRO-Infinity can
support operators of arbitrary sizes, without refactoring for model
parallelism to fit them in limited GPU memory.


.. autoclass:: deepscale.zero.TiledLinear
    :members:
