# coding=utf-8
# Copyright (c) 2024, The KhulnaSoft DeepScale Team. All rights reserved.
#
# Note: please copy webtext data to "Megatron-LM" folder, before running this script.

import sys
import unittest
import os

sys.path.append("../DeepScaleExamples/Megatron_GPT2")
sys.path.append("../DeepScaleExamples/BingBertSquad")

# Import the test cases here.
import Megatron_GPT2
import BingBertSquad

def pytest_hack(runner_result):
    """This is an ugly hack to get the unittest suites to play nicely with
    pytest. Otherwise failed tests are not reported by pytest for some reason.

    Long-term, these model tests should be adapted to pytest.
    """
    if not runner_result.wasSuccessful():
        print("SUITE UNSUCCESSFUL:", file=sys.stderr)
        for fails in runner_result.failures:
            print(fails, file=sys.stderr)
        assert runner_result.wasSuccessful()  # fail the test

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        return False
    return True

def check_loss_in_file(filepath):
    if not check_file_exists(filepath):
        return False
    with open(filepath, 'r') as f:
        content = f.read()
        if 'loss' not in content:
            print(f"No loss found in file {filepath}", file=sys.stderr)
            return False
    return True

def test_megatron():
    runner = unittest.TextTestRunner(failfast=True)
    result = runner.run(Megatron_GPT2.suite())
    # Check files
    baseline_log = './baseline/gpt2_func_mp1_gpu2_node1_bs8_step1000_layer2_hidden128_seq64_head8.log'
    test_log = './test/gpt2_func_mp1_gpu2_node1_bs8_step1000_layer2_hidden128_seq64_head8_ds-20240710-195711.log'
    check_loss_in_file(baseline_log)
    check_loss_in_file(test_log)
    pytest_hack(result)

def test_megatron_checkpoint():
    runner = unittest.TextTestRunner(failfast=True)
    result = runner.run(Megatron_GPT2.checkpoint_suite())
    checkpoint_file = 'ckpt_mp2_gpu8_w_zero1/latest_checkpointed_iteration.txt'
    if not check_file_exists(checkpoint_file):
        print(f"Checkpoint file not created: {checkpoint_file}", file=sys.stderr)
    pytest_hack(result)

def test_squad():
    runner = unittest.TextTestRunner(failfast=True)
    result = runner.run(BingBertSquad.suite())
    # Check files
    baseline_log = './baseline/BingBertSquad_func_gpu4__fp16--print_steps1--max_steps8--max_steps_per_epoch4.log'
    test_log = './test/BingBertSquad_func_gpu4__fp16--print_steps1--max_steps8--max_steps_per_epoch4_ds-20240710-195711.log'
    check_loss_in_file(baseline_log)
    check_loss_in_file(test_log)
    pytest_hack(result)

if __name__ == "__main__":
    test_megatron()
    test_megatron_checkpoint()
    test_squad()
