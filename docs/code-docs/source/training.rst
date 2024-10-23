Training API
============

:func:`deepscale.initialize` returns a *training engine* in its first argument
of type :class:`DeepScaleEngine`. This engine is used to progress training:

.. code-block:: python

    for step, batch in enumerate(data_loader):
        #forward() method
        loss = model_engine(batch)

        #runs backpropagation
        model_engine.backward(loss)

        #weight update
        model_engine.step()

Forward Propagation
-------------------
.. autofunction:: deepscale.DeepScaleEngine.forward

Backward Propagation
--------------------
.. autofunction:: deepscale.DeepScaleEngine.backward

Optimizer Step
--------------
.. autofunction:: deepscale.DeepScaleEngine.step

Gradient Accumulation
---------------------
.. autofunction:: deepscale.DeepScaleEngine.is_gradient_accumulation_boundary


Model Saving
------------
.. autofunction:: deepscale.DeepScaleEngine.save_16bit_model


Additionally when a DeepScale checkpoint is created, a script ``zero_to_fp32.py`` is added there which can be used to reconstruct fp32 master weights into a single pytorch ``state_dict`` file.
