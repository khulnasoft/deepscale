# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import argparse
import pytest
import deepscale
from deepscale.utils.numa import parse_range_list


def basic_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int)
    return parser


def test_no_ds_arguments_no_ds_parser():
    parser = basic_parser()
    args = parser.parse_args(['--num_epochs', '2'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert not hasattr(args, 'deepscale')
    assert not hasattr(args, 'deepscale_config')


def test_no_ds_arguments():
    parser = basic_parser()
    parser = deepscale.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepscale')
    assert args.deepscale == False

    assert hasattr(args, 'deepscale_config')
    assert args.deepscale_config is None


def test_no_ds_enable_argument():
    parser = basic_parser()
    parser = deepscale.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2', '--deepscale_config', 'foo.json'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepscale')
    assert args.deepscale == False

    assert hasattr(args, 'deepscale_config')
    assert type(args.deepscale_config) == str
    assert args.deepscale_config == 'foo.json'


def test_no_ds_config_argument():
    parser = basic_parser()
    parser = deepscale.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2', '--deepscale'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepscale')
    assert type(args.deepscale) == bool
    assert args.deepscale == True

    assert hasattr(args, 'deepscale_config')
    assert args.deepscale_config is None


def test_no_ds_parser():
    parser = basic_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args(['--num_epochs', '2', '--deepscale'])


def test_core_deepscale_arguments():
    parser = basic_parser()
    parser = deepscale.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2', '--deepscale', '--deepscale_config', 'foo.json'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepscale')
    assert type(args.deepscale) == bool
    assert args.deepscale == True

    assert hasattr(args, 'deepscale_config')
    assert type(args.deepscale_config) == str
    assert args.deepscale_config == 'foo.json'


def test_core_binding_arguments():
    core_list = parse_range_list("0,2-4,6,8-9")
    assert core_list == [0, 2, 3, 4, 6, 8, 9]

    try:
        # negative case for range overlapping
        core_list = parse_range_list("0,2-6,5-9")
    except ValueError as e:
        pass
    else:
        # invalid core list must fail
        assert False

    try:
        # negative case for reverse order -- case 1
        core_list = parse_range_list("8,2-6")
    except ValueError as e:
        pass
    else:
        # invalid core list must fail
        assert False

    try:
        # negative case for reverse order -- case 2
        core_list = parse_range_list("1,6-2")
    except ValueError as e:
        pass
    else:
        # invalid core list must fail
        assert False
