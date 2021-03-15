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

import argparse
import os

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from bayesian_aggregation.bayesian_aggregation.clv_model import CLVModel
from bayesian_aggregation.scripts.ba_settings_data_gp_iclr2021 import (
    get_settings_and_data,
)
from bayesian_aggregation.scripts.evaluation import (
    compute_predictive_likelihood,
    plot_prediction,
)

# parse command line arguments
parser = argparse.ArgumentParser(
    description="Run experiments for ICLR2021 submission: "
    "Volpp et al., 'Bayesian Context Aggregation for Neural Processes'."
)
parser.add_argument(
    "experiment",
    type=str,
    help="the name of the experiment",
    choices=["RBFGP", "WeaklyPeriodicGP", "Matern52GP"],
)
parser.add_argument(
    "aggregator", type=str, help="the aggregator to use", choices=["BA", "MA"]
)
parser.add_argument(
    "likelihood_approximation",
    type=str,
    help="the likelihood approximation to use",
    choices=["PB", "VI", "MC"],
)
args = parser.parse_args()
experiment = args.experiment
aggregator = args.aggregator
likelihood_approximation = args.likelihood_approximation

# settings
settings, dataloader_m, dataloader_v, dataloader_t = get_settings_and_data(
    benchmark_name=experiment,
    aggregator=aggregator,
    likelihood_approximation=likelihood_approximation,
)

# train model for 200 epochs
logpath = "./log/{}_{}_{}".format(experiment, aggregator, likelihood_approximation)
try:
    os.makedirs(logpath, exist_ok=False)
    model = CLVModel(logpath=logpath)
    model.initialize_new_model(settings=settings)
except FileExistsError:
    print("Logpath exists, using existing weights!")
    model = CLVModel(logpath=logpath)
    model.load_model(load_epoch="last")

model.meta_fit(
    dataloader_m=dataloader_m,
    dataloader_v=dataloader_v,
    max_epochs=200,
    tb_writer=SummaryWriter(log_dir=logpath),
)

# evaluate model on test data
plot_prediction(model, dataloader_t, n_tsk=3, n_ctx=20)
ll = compute_predictive_likelihood(model, dataloader_t, n_ctx_max=64)
print("\n************* RESULTS ************ ")
print("Experiment                = {}".format(experiment))
print("Aggregator                = {}".format(aggregator))
print("Likelihood Approximation  = {}".format(likelihood_approximation))
print("Predictive log-likelihood = {:.2f}".format(ll))
print("********************************** ")
plt.show()
