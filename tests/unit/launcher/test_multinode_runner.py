# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from copy import deepcopy
from deepscale.launcher import multinode_runner as mnrunner
from deepscale.launcher.runner import encode_world_info, parse_args
import os
import pytest


@pytest.fixture
def runner_info():
    hosts = {'worker-0': 4, 'worker-1': 4}
    world_info = encode_world_info(hosts)
    env = deepcopy(os.environ)
    args = parse_args(['test_launcher.py'])
    return env, hosts, world_info, args


def test_pdsh_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.PDSHRunner(args, world_info)
    cmd, kill_cmd, env = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'pdsh'
    assert env['PDSH_RCMD_TYPE'] == 'ssh'


def test_openmpi_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'
    assert 'eth0' in cmd


def test_btl_nic_openmpi_runner(runner_info):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(['--launcher_arg', '-mca btl_tcp_if_include eth1', 'test_launcher.py'])

    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert 'eth0' not in cmd
    assert 'eth1' in cmd


def test_btl_nic_two_dashes_openmpi_runner(runner_info):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(['--launcher_arg', '--mca btl_tcp_if_include eth1', 'test_launcher.py'])

    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert 'eth0' not in cmd
    assert 'eth1' in cmd


def test_mpich_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.MPICHRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'


def test_slurm_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.SlurmRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'srun'


def test_mvapich_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.MVAPICHRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'
