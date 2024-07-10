Model Checkpointing
===================

DeepScale provides routines for checkpointing model state during training.

Loading Training Checkpoints
----------------------------
.. autofunction:: deepscale.DeepScaleEngine.load_checkpoint

Saving Training Checkpoints
---------------------------
.. autofunction:: deepscale.DeepScaleEngine.save_checkpoint


ZeRO Checkpoint fp32 Weights Recovery
-------------------------------------

DeepScale provides routines for extracting fp32 weights from the saved ZeRO checkpoint's optimizer states.

.. autofunction:: deepscale.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint

.. autofunction:: deepscale.utils.zero_to_fp32.load_state_dict_from_zero_checkpoint

.. autofunction:: deepscale.utils.zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict
