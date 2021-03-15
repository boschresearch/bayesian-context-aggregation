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

# ******************************************************************
# mlp.py
# A simple multi-layer perceptron in pytorch.
# ******************************************************************

import os
import pickle as pkl

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.container import ModuleList


def tanh2(x, min_y, max_y):
    scale_x = 1 / ((max_y - min_y) / 2)
    return (max_y - min_y) / 2 * (torch.tanh(x * scale_x) + 1.0) + min_y


class MLP(nn.Module):
    def __init__(
        self,
        name,
        logpath=None,
        epoch=None,
        d_in=None,
        d_out=None,
        mlp_layers=None,
        shape=None,
        d_avg=None,
        n_hidden=None,
        f_act=None,
        f_out=(None, {}),
        seed=None,
        use_standard_initialization=True,
        std_weight=None,
        std_bias=None,
    ):
        super(MLP, self).__init__()

        self.name = name

        self.function_name_mappings = {
            "tanh": torch.tanh,
            "tanh2": tanh2,
            "relu": torch.relu,
            "softplus": F.softplus,
            "exp": torch.exp,
            "None": None,
        }

        if logpath is not None:
            assert d_in is None
            assert d_out is None
            assert mlp_layers is None
            assert f_act is None
            assert f_out == (None, {})
            assert seed is None
            self.load_parameters(logpath)
            self.create_network()
            self.load_weights(logpath, epoch=epoch)
        else:
            assert epoch is None
            self.d_in = d_in
            self.d_out = d_out
            if mlp_layers is not None:
                # either provide the arch spec directly via mlp_layers
                assert shape is None
                assert d_avg is None
                assert n_hidden is None
                self.arch_spec = mlp_layers
            else:
                # or compute the arch spec from shape, d_avg, and n_hidden
                assert mlp_layers is None
                self.arch_spec = self.compute_arch_spec(
                    shape=shape, d_avg=d_avg, n_hidden=n_hidden
                )
            self.f_act = self.function_name_mappings[f_act]
            self.out_trafo_fun = (
                self.function_name_mappings[f_out[0]] if f_out[0] is not None else None
            )
            self.out_trafo_params = f_out[1]
            self.seed = seed
            self.create_network()
            if not use_standard_initialization:
                self.initialize_weights(std_weight=std_weight, std_bias=std_bias)

    def create_network(self):
        # process architecture
        self.n_hidden_layers = len(self.arch_spec)
        self.is_linear = self.n_hidden_layers == 0  # no hidden layers --> linear model
        if not self.is_linear:
            assert self.f_act is not None

        # seeding
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # define the network
        if self.is_linear:
            self.linears = ModuleList(
                [nn.Linear(in_features=self.d_in, out_features=self.d_out)]
            )
        else:
            self.linears = ModuleList(
                [nn.Linear(in_features=self.d_in, out_features=self.arch_spec[0])]
            )
            for i in range(1, len(self.arch_spec)):
                self.linears.append(
                    nn.Linear(
                        in_features=self.linears[-1].out_features,
                        out_features=self.arch_spec[i],
                    )
                )
            self.linears.append(
                nn.Linear(
                    in_features=self.linears[-1].out_features, out_features=self.d_out
                )
            )

    def forward(self, X, output_layer=None):
        if output_layer is None:
            output_layer = self.n_hidden_layers + 1
        assert 0 <= output_layer <= len(self.arch_spec) + 1

        Y = X
        if output_layer == 0:
            return Y

        if self.is_linear:
            Y = (
                self.linears[0](Y)
                if self.out_trafo_fun is None
                else self.out_trafo_fun(self.linears[0](Y), **self.out_trafo_params)
            )
        else:
            # do not iterate directly over self.linears, this is slow using ModuleList
            for i in range(self.n_hidden_layers):
                Y = self.f_act(self.linears[i](Y))
                if i + 1 == output_layer:
                    return Y
            Y = (
                self.linears[-1](Y)
                if self.out_trafo_fun is None
                else self.out_trafo_fun(self.linears[-1](Y), **self.out_trafo_params)
            )

        return Y

    def save(self, path, epoch):
        with open(os.path.join(path, self.name + "_parameters.pkl"), "wb") as f:
            parameters = {
                "d_in": self.d_in,
                "d_out": self.d_out,
                "arch_spec": self.arch_spec,
                "f_act": self.f_act.__name__ if self.f_act is not None else "None",
                "seed": self.seed,
                "out_trafo_fun": self.out_trafo_fun.__name__
                if self.out_trafo_fun is not None
                else "None",
                "out_trafo_params": self.out_trafo_params
                if self.out_trafo_params is not None
                else "None",
            }
            pkl.dump(parameters, f)

        if epoch is not None:
            with open(
                os.path.join(path, self.name + "_weights_{:d}".format(epoch)), "wb"
            ) as f:
                torch.save(self.state_dict(), f)
        else:
            with open(os.path.join(path, self.name + "_weights"), "wb") as f:
                torch.save(self.state_dict(), f)

    def delete_all_weight_files(self, path):
        for file in os.listdir(path):
            if file.startswith(self.name + "_weights"):
                os.remove(os.path.join(path, file))

    def load_parameters(self, path):
        with open(os.path.join(path, self.name + "_parameters.pkl"), "rb") as f:
            parameters = pkl.load(f)
        self.d_in = parameters["d_in"]
        self.d_out = parameters["d_out"]
        self.arch_spec = parameters["arch_spec"]
        self.seed = parameters["seed"]
        self.f_act = self.function_name_mappings[parameters["f_act"]]
        self.out_trafo_fun = self.function_name_mappings[parameters["out_trafo_fun"]]

    def load_weights(self, path, epoch):
        if epoch is not None:
            self.load_state_dict(
                torch.load(
                    os.path.join(path, self.name + "_weights_{:d}".format(epoch))
                )
            )
        else:
            self.load_state_dict(torch.load(os.path.join(path, self.name + "_weights")))
