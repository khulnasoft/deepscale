#!/bin/bash

deepscale test_mics_config.py --mics_shard_size=1

deepscale test_mics_config.py --mics_shard_size=2

# for debugging the hierarchical params gathering
export NDEV_PER_NODE=2
deepscale test_mics_config.py --mics_shard_size=4 --mics_hierarchical_params_gather
