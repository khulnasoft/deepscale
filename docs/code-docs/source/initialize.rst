Training Setup
==============

.. _deepscale-args:

Argument Parsing
----------------
DeepScale uses the `argparse <https://docs.python.org/3/library/argparse.html>`_ library to
supply commandline configuration to the DeepScale runtime. Use ``deepscale.add_config_arguments()``
to add DeepScale's builtin arguments to your application's parser.

.. code-block:: python

    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepScale configuration arguments
    parser = deepscale.add_config_arguments(parser)
    cmd_args = parser.parse_args()

.. autofunction:: deepscale.add_config_arguments


.. _deepscale-init:

Training Initialization
-----------------------
The entrypoint for all training with DeepScale is ``deepscale.initialize()``. Will initialize distributed backend if it is not intialized already.

Example usage:

.. code-block:: python

    model_engine, optimizer, _, _ = deepscale.initialize(args=cmd_args,
                                                         model=net,
                                                         model_parameters=net.parameters())

.. autofunction:: deepscale.initialize

Distributed Initialization
-----------------------
Optional distributed backend initializating separate from ``deepscale.initialize()``. Useful in scenarios where the user wants to use torch distributed calls before calling ``deepscale.initialize()``, such as when using model parallelism, pipeline parallelism, or certain data loader scenarios.

.. autofunction:: deepscale.init_distributed
