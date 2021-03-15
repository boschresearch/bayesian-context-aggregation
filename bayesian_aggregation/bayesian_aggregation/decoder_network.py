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


class DecoderNetworkPB:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_z = kwargs["d_z"]
        self.arch = kwargs["arch"]

        # process network shapes
        self.mlp_layers_mu_y = kwargs["mlp_layers_mu_y"]
        self.mlp_layers_std_y = kwargs["mlp_layers_std_y"]

        self.f_act = kwargs["f_act"]
        self.safe_log = kwargs["safe_log"]
        self.seed = kwargs["seed"]

        self.mu_y_net = self.std_y_net = self.mu_y_std_y_net = None
        self.create_networks()

    def set_device(self, device):
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            self.mu_y_net = self.mu_y_net.to(device)
            self.std_y_net = self.std_y_net.to(device)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = self.mu_y_std_y_net.to(device)

    def init_weights(self):
        pass  # pytorch takes care of this

    def save_weights(self, logpath, epoch):
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            self.mu_y_net.save(path=logpath, epoch=epoch)
            self.std_y_net.save(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net.save(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            self.mu_y_net.load_weights(path=logpath, epoch=epoch)
            self.std_y_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        if self.arch == "separate_networks_combined_input":
            self.mu_y_net = MLP(
                name="decoder_mu",
                d_in=self.d_x + self.d_z * 2,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_mu_y,
                f_act=self.f_act,
                seed=self.seed,
            )
            self.std_y_net = MLP(
                name="decoder_cov",
                d_in=self.d_x + self.d_z * 2,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_std_y,
                f_act=self.f_act,
                seed=self.seed,
            )
        elif self.arch == "separate_networks_separate_input":
            self.mu_y_net = MLP(
                name="decoder_mu",
                d_in=self.d_x + self.d_z,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_mu_y,
                f_act=self.f_act,
                seed=self.seed,
            )
            self.std_y_net = MLP(
                name="decoder_cov",
                d_in=self.d_x + self.d_z,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_std_y,
                f_act=self.f_act,
                seed=self.seed,
            )
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = MLP(
                name="decoder_mu_cov",
                d_in=self.d_x + self.d_z * 2,
                d_out=2 * self.d_y,
                mlp_layers=self.mlp_layers_mu_y,  # ignore self.mlp_layers_std_y
                f_act=self.f_act,
                seed=self.seed,
            )
        else:
            raise ValueError("Unknown network type: {}!".format(self.arch))

    def decode(self, x, mu_z, cov_z):
        assert x.ndim == 3
        assert mu_z.ndim == cov_z.ndim == 3

        n_tsk = mu_z.shape[0]
        n_ls = mu_z.shape[1]
        n_tst = x.shape[1]

        # covariance parametrization
        cov_z = self.parametrize_latent_cov(cov_z)

        # add latent-state-wise batch dimension to X
        x = x[:, None, :, :]
        x = x.expand(n_tsk, n_ls, n_tst, self.d_x)

        # add dataset-wise batch dimension to latent states
        mu_z = mu_z[:, :, None, :]
        cov_z = cov_z[:, :, None, :]

        # prepare input to decoder network
        if self.arch == "separate_networks_combined_input":
            mu_z_cov_z = torch.cat((mu_z, cov_z), dim=3)
            mu_z_cov_z = mu_z_cov_z.expand((n_tsk, n_ls, n_tst, self.d_z * 2))
            input_mu = input_std = torch.cat((x, mu_z_cov_z), dim=3)
        elif self.arch == "separate_networks_separate_input":
            mu_z = mu_z.expand((n_tsk, n_ls, n_tst, self.d_z))
            cov_z = cov_z.expand((n_tsk, n_ls, n_tst, self.d_z))
            input_mu = torch.cat((x, mu_z), dim=3)
            input_std = torch.cat((x, cov_z), dim=3)
        elif self.arch == "two_heads":
            mu_z_cov_z = torch.cat((mu_z, cov_z), dim=3)
            mu_z_cov_z = mu_z_cov_z.expand((n_tsk, n_ls, n_tst, self.d_z * 2))
            input_two_head = torch.cat((x, mu_z_cov_z), dim=3)

        # decode
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            mu_y = self.mu_y_net(input_mu)
            std_y = self.std_y_net(input_std)
        elif self.arch == "two_heads":
            mu_y_std_y = self.mu_y_std_y_net(input_two_head)
            mu_y = mu_y_std_y[:, : self.d_y]
            std_y = mu_y_std_y[:, self.d_y :]

        # deparametrize
        std_y = torch.exp(std_y)

        return mu_y, std_y

    def parametrize_latent_cov(self, cov):
        cov = cov + self.safe_log
        parametrized_cov = torch.log(cov)

        return parametrized_cov

    @property
    def parameters(self):
        params = []
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            params += list(self.mu_y_net.parameters()) + list(
                self.std_y_net.parameters()
            )
        elif self.arch == "two_heads":
            params += list(self.mu_y_std_y_net.parameters())
        return params


class DecoderNetworkSamples:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_z = kwargs["d_z"]
        self.arch = kwargs["arch"]

        # process network shapes
        self.mlp_layers_mu_y = kwargs["mlp_layers_mu_y"]
        self.mlp_layers_std_y = kwargs["mlp_layers_std_y"]

        self.f_act = kwargs["f_act"]
        self.seed = kwargs["seed"]

        self.mu_y_net = self.std_y_net = self.mu_y_std_y_net = None
        self.create_networks()

    def set_device(self, device):
        if self.arch == "separate_networks":
            self.mu_y_net = self.mu_y_net.to(device)
            self.std_y_net = self.std_y_net.to(device)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = self.mu_y_std_y_net.to(device)

    def init_weights(self):
        pass  # pytorch takes care of this

    def save_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.mu_y_net.save(path=logpath, epoch=epoch)
            self.std_y_net.save(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net.save(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.mu_y_net.load_weights(path=logpath, epoch=epoch)
            self.std_y_net.load_weights(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        if self.arch == "separate_networks":
            self.mu_y_net = MLP(
                name="decoder_mu",
                d_in=self.d_x + self.d_z,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_mu_y,
                f_act=self.f_act,
                seed=self.seed,
            )
            self.std_y_net = MLP(
                name="decoder_cov",
                d_in=self.d_x + self.d_z,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_std_y,
                f_act=self.f_act,
                seed=self.seed,
            )
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = MLP(
                name="decoder_mu_cov",
                d_in=self.d_x + self.d_z,
                d_out=2 * self.d_y,
                mlp_layers=self.mlp_layers_mu_y,  # ignore mlp_layers_std_y
                f_act=self.f_act,
                seed=self.seed,
            )
        else:
            raise ValueError("Unknown encoder network type!")

    def decode(self, x, z):
        assert x.ndim == 3  # (n_tsk, n_tst, d_x)
        assert z.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)

        n_tsk = z.shape[0]
        n_ls = z.shape[1]
        n_marg = z.shape[2]
        n_tst = x.shape[1]

        # add latent-state-wise batch dimension to x
        x = x[:, None, None, :, :]
        x = x.expand(n_tsk, n_ls, n_marg, n_tst, self.d_x)

        # add dataset-wise batch dimension to sample
        z = z[:, :, :, None, :]
        z = z.expand((n_tsk, n_ls, n_marg, n_tst, self.d_z))

        # prepare input to decoder network
        input = torch.cat((x, z), dim=4)

        # decode
        if self.arch == "separate_networks":
            mu_y = self.mu_y_net(input)
            std_y = self.std_y_net(input)
        elif self.arch == "two_heads":
            mu_cov_y = self.mu_y_std_y_net(input)
            mu_y = mu_cov_y[:, : self.d_y]
            std_y = mu_cov_y[:, self.d_y :]

        # deparametrize
        std_y = torch.exp(std_y)

        return mu_y, std_y

    @property
    def parameters(self):
        params = []
        if self.arch == "separate_networks":
            params += list(self.mu_y_net.parameters())
            params += list(self.std_y_net.parameters())
        elif self.arch == "two_heads":
            params += list(self.mu_y_std_y_net.parameters())
        return params
