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


import torch

from bayesian_aggregation.bayesian_aggregation.mlp import MLP


class EncoderNetworkBA:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_lo = kwargs["d_lo"]
        self.arch = kwargs["arch"]
        self.f_act = kwargs["f_act"]
        self.seed = kwargs["seed"]

        # process network shapes
        self.mlp_layers_r = kwargs["mlp_layers_r"]
        self.mlp_layers_cov_r = kwargs["mlp_layers_cov_r"]

        self.r_net = self.cov_r_net = self.r_cov_r_net = None
        self.create_networks()

    def set_device(self, device):
        if self.arch == "separate_networks":
            self.r_net = self.r_net.to(device)
            self.cov_r_net = self.cov_r_net.to(device)
        elif self.arch == "two_heads":
            self.r_cov_r_net = self.r_cov_r_net.to(device)

    def init_weights(self):
        pass

    def save_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.r_net.save(path=logpath, epoch=epoch)
            self.cov_r_net.save(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.r_cov_r_net.save(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.r_net.load_weights(path=logpath, epoch=epoch)
            self.cov_r_net.load_weights(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.r_cov_r_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        if self.arch == "separate_networks":
            self.r_net = MLP(
                name="encoder_mu",
                d_in=self.d_x + self.d_y,
                d_out=self.d_lo,
                mlp_layers=self.mlp_layers_r,
                f_act=self.f_act,
                seed=self.seed,
            )
            self.cov_r_net = MLP(
                name="encoder_cov",
                d_in=self.d_x + self.d_y,
                d_out=self.d_lo,
                mlp_layers=self.mlp_layers_cov_r,
                f_act=self.f_act,
                seed=self.seed,
            )
        elif self.arch == "two_heads":
            self.r_cov_r_net = MLP(
                name="encoder_mu_cov",
                d_in=self.d_x + self.d_y,
                d_out=2 * self.d_lo,
                mlp_layers=self.mlp_layers_r,  # ignore self.mlp_layers_cov_r
                f_act=self.f_act,
                seed=self.seed,
            )
        else:
            raise ValueError("Unknown encoder network type!")

    def encode(self, x, y):
        assert x.ndim == y.ndim == 3

        # prepare input to encoder network
        encoder_input = torch.cat((x, y), dim=2)

        # encode
        if self.arch == "separate_networks":
            r, cov_r = self.r_net(encoder_input), self.cov_r_net(encoder_input)
        elif self.arch == "two_heads":
            mu_r_cov_r = self.r_cov_r_net(encoder_input)
            r = mu_r_cov_r[:, :, : self.d_lo]
            cov_r = mu_r_cov_r[:, :, self.d_lo :]

        cov_r = torch.exp(cov_r)

        return r, cov_r

    @property
    def parameters(self):
        if self.arch == "separate_networks":
            return list(self.r_net.parameters()) + list(self.cov_r_net.parameters())
        elif self.arch == "two_heads":
            return list(self.r_cov_r_net.parameters())


class EncoderNetworkMA:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_lo = kwargs["d_lo"]
        self.mlp_layers = kwargs["mlp_layers"]
        self.f_act = kwargs["f_act"]
        self.seed = kwargs["seed"]

        self.r_net = None
        self.create_networks()

    def set_device(self, device):
        self.r_net = self.r_net.to(device)

    def init_weights(self):
        pass

    def save_weights(self, logpath, epoch):
        self.r_net.save(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        self.r_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        self.r_net = MLP(
            name="encoder_r",
            d_in=self.d_x + self.d_y,
            d_out=self.d_lo,
            mlp_layers=self.mlp_layers,
            f_act=self.f_act,
            seed=self.seed,
        )

    def encode(self, x, y):
        assert x.ndim == y.ndim == 3

        # prepare input to encoder network
        encoder_input = torch.cat((x, y), dim=2)

        # encode
        r = self.r_net(encoder_input)

        return (r,)

    @property
    def parameters(self):
        return list(self.r_net.parameters())
