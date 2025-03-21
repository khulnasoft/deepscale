# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

from deepscale.monitor.tensorboard import TensorBoardMonitor
from deepscale.monitor.wandb import WandbMonitor
from deepscale.monitor.csv_monitor import csvMonitor
from deepscale.monitor.config import DeepScaleMonitorConfig
from deepscale.monitor.comet import CometMonitor

from unit.common import DistributedTest
from unittest.mock import Mock, patch
from deepscale.runtime.config import DeepScaleConfig

import deepscale.comm as dist


class TestTensorBoard(DistributedTest):
    world_size = 2

    def test_tensorboard(self):
        config_dict = {
            "train_batch_size": 2,
            "tensorboard": {
                "enabled": True,
                "output_path": "test_output/ds_logs/",
                "job_name": "test"
            }
        }
        ds_config = DeepScaleConfig(config_dict)
        tb_monitor = TensorBoardMonitor(ds_config.monitor_config.tensorboard)
        assert tb_monitor.enabled == True
        assert tb_monitor.output_path == "test_output/ds_logs/"
        assert tb_monitor.job_name == "test"

    def test_empty_tensorboard(self):
        config_dict = {"train_batch_size": 2, "tensorboard": {}}
        ds_config = DeepScaleConfig(config_dict)
        tb_monitor = TensorBoardMonitor(ds_config.monitor_config.tensorboard)
        defaults = DeepScaleMonitorConfig().tensorboard
        assert tb_monitor.enabled == defaults.enabled
        assert tb_monitor.output_path == defaults.output_path
        assert tb_monitor.job_name == defaults.job_name


class TestWandB(DistributedTest):
    world_size = 2

    def test_wandb(self):
        config_dict = {
            "train_batch_size": 2,
            "wandb": {
                "enabled": False,
                "group": "my_group",
                "team": "my_team",
                "project": "my_project"
            }
        }
        ds_config = DeepScaleConfig(config_dict)
        wandb_monitor = WandbMonitor(ds_config.monitor_config.wandb)
        assert wandb_monitor.enabled == False
        assert wandb_monitor.group == "my_group"
        assert wandb_monitor.team == "my_team"
        assert wandb_monitor.project == "my_project"

    def test_empty_wandb(self):
        config_dict = {"train_batch_size": 2, "wandb": {}}
        ds_config = DeepScaleConfig(config_dict)
        wandb_monitor = WandbMonitor(ds_config.monitor_config.wandb)
        defaults = DeepScaleMonitorConfig().wandb
        assert wandb_monitor.enabled == defaults.enabled
        assert wandb_monitor.group == defaults.group
        assert wandb_monitor.team == defaults.team
        assert wandb_monitor.project == defaults.project


class TestCSVMonitor(DistributedTest):
    world_size = 2

    def test_csv_monitor(self):
        config_dict = {
            "train_batch_size": 2,
            "csv_monitor": {
                "enabled": True,
                "output_path": "test_output/ds_logs/",
                "job_name": "test"
            }
        }
        ds_config = DeepScaleConfig(config_dict)
        csv_monitor = csvMonitor(ds_config.monitor_config.csv_monitor)
        assert csv_monitor.enabled == True
        assert csv_monitor.output_path == "test_output/ds_logs/"
        assert csv_monitor.job_name == "test"

    def test_empty_csv_monitor(self):
        config_dict = {"train_batch_size": 2, "csv_monitor": {}}
        ds_config = DeepScaleConfig(config_dict)
        csv_monitor = csvMonitor(ds_config.monitor_config.csv_monitor)
        defaults = DeepScaleMonitorConfig().csv_monitor
        assert csv_monitor.enabled == defaults.enabled
        assert csv_monitor.output_path == defaults.output_path
        assert csv_monitor.job_name == defaults.job_name


class TestCometMonitor(DistributedTest):
    world_size = 2

    def test_comet_monitor(self):
        import comet_ml
        mock_experiment = Mock()
        mock_start = Mock(return_value=mock_experiment)

        config_dict = {
            "train_batch_size": 2,
            "comet": {
                "enabled": True,
                "samples_log_interval": 42,
                "workspace": "some-workspace",
                "project": "some-project",
                "api_key": "some-api-key",
                "experiment_name": "some-experiment-name",
                "experiment_key": "some-experiment-key",
                "mode": "get_or_create",
                "online": True
            }
        }

        ds_config = DeepScaleConfig(config_dict)

        with patch.object(comet_ml, "start", mock_start):
            comet_monitor = CometMonitor(ds_config.monitor_config.comet)

        assert comet_monitor.enabled is True
        assert comet_monitor.samples_log_interval == 42

        # experiment should be initialized via comet_ml.start only if rank == 0
        if dist.get_rank() == 0:
            mock_start.assert_called_once_with(
                api_key="some-api-key",
                project="some-project",
                workspace="some-workspace",
                experiment_key="some-experiment-key",
                mode="get_or_create",
                online=True,
            )

            mock_experiment.set_name.assert_called_once_with("some-experiment-name")
            assert comet_monitor.experiment is mock_experiment
        else:
            mock_start.assert_not_called()

    def test_empty_comet(self):
        import comet_ml
        mock_start = Mock()

        config_dict = {"train_batch_size": 2, "comet": {}}
        ds_config = DeepScaleConfig(config_dict)

        with patch.object(comet_ml, "start", mock_start):
            comet_monitor = CometMonitor(ds_config.monitor_config.comet)

        defaults = DeepScaleMonitorConfig().comet
        assert comet_monitor.enabled == defaults.enabled
        assert comet_monitor.samples_log_interval == defaults.samples_log_interval
        mock_start.assert_not_called()
