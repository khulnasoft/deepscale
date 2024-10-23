Inference Setup
-----------------------
The entrypoint for inference with DeepScale is ``deepscale.init_inference()``.

Example usage:

.. code-block:: python

    engine = deepscale.init_inference(model=net, config=config)

The ``DeepScaleInferenceConfig`` is used to control all aspects of initializing
the ``InferenceEngine``. The config should be passed as a dictionary to
``init_inference``, but parameters can also be passed as keyword arguments.

.. _DeepScaleInferenceConfig:
.. autopydantic_model:: deepscale.inference.config.DeepScaleInferenceConfig

.. _DeepScaleTPConfig:
.. autopydantic_model:: deepscale.inference.config.DeepScaleTPConfig

.. _DeepScaleMoEConfig:
.. autopydantic_model:: deepscale.inference.config.DeepScaleMoEConfig

.. _QuantizationConfig:
.. autopydantic_model:: deepscale.inference.config.QuantizationConfig

.. _InferenceCheckpointConfig:
.. autopydantic_model:: deepscale.inference.config.InferenceCheckpointConfig


Example config:

.. code-block:: python

    config = {
	"kernel_inject": True,
	"tensor_parallel": {"tp_size": 4},
	"dtype": "fp16",
	"enable_cuda_graph": False
    }

.. autofunction:: deepscale.init_inference
