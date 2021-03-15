# Copyright (c) 2020 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

import yaml
from torch.utils.data import DataLoader

from bayesian_aggregation.data.dataset import NPIterableDataset


def get_settings_and_data(benchmark_name, aggregator, likelihood_approximation):
    assert benchmark_name in ["RBFGP", "WeaklyPeriodicGP", "Matern52GP"]
    assert aggregator in ["BA", "MA"]
    assert likelihood_approximation in ["PB", "VI", "MC"]

    # load settings determined by HPO
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "../settings_iclr2021")
    fn = (
        "settings_" + benchmark_name + "_" + aggregator + "_" + likelihood_approximation
    )
    fn += ".yaml"

    with open(os.path.join(path, fn), "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # data
    n_task_m = 256 * 16
    n_data_per_task_m = 128
    n_task_v = 256
    n_data_per_task_v = 128
    n_task_t = 256
    n_data_per_task_t = 256 + 64
    dataloader_m = DataLoader(
        dataset=NPIterableDataset(
            benchmark_name=benchmark_name,
            n_task=n_task_m,
            n_data_per_task=n_data_per_task_m,
            seed=settings["seed"],
        ),
        batch_size=settings["training"]["batch_size"],
    )
    dataloader_v = DataLoader(
        dataset=NPIterableDataset(
            benchmark_name=benchmark_name,
            n_task=n_task_v,
            n_data_per_task=n_data_per_task_v,
            seed=2 * settings["seed"],
        ),
        batch_size=settings["training"]["batch_size"],
    )
    dataloader_t = DataLoader(
        dataset=NPIterableDataset(
            benchmark_name=benchmark_name,
            n_task=n_task_t,
            n_data_per_task=n_data_per_task_t,
            seed=3 * settings["seed"],
        ),
        batch_size=settings["training"]["batch_size"],
    )

    return settings, dataloader_m, dataloader_v, dataloader_t
