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


class BayesianAggregator:
    def __init__(self, **kwargs):
        self.d_z = self.d_lo = kwargs["d_z"]
        self.mu_z_init_scalar = kwargs["init_data"][0]
        self.var_z_init_scalar = kwargs["init_data"][1]

        self.mu_z_init = None
        self.cov_z_init = None
        self.mu_z = None
        self.cov_z = None

        self.n_tsk = None

        self.device = None

    @property
    def agg_state(self):
        return self.mu_z, self.cov_z

    @property
    def initial_agg_state(self):
        return self.mu_z_init[:, 0, :], self.cov_z_init[:, 0, :]

    @property
    def last_agg_state(self):
        return self.mu_z[:, -1, :], self.cov_z[:, -1, :]

    @property
    def latent_state(self):
        return self.agg2latent(self.agg_state)

    @property
    def initial_latent_state(self):
        return self.agg2latent(self.initial_agg_state)

    @property
    def last_latent_state(self):
        return self.agg2latent(self.last_agg_state)

    @property
    def parameters(self):
        return []

    def init_weights(self):
        pass

    def save_weights(self, logpath, epoch):
        pass

    def load_weights(self, logpath, epoch):
        pass

    def delete_all_weight_files(self, logpath):
        pass

    def set_device(self, device):
        self.device = device

    def _append_agg_state(self, agg_state):
        mu_z = agg_state[0]
        cov_z = agg_state[1]

        assert mu_z.ndim == cov_z.ndim == 2
        assert mu_z.shape == cov_z.shape

        # add ls dimension
        mu_z = mu_z[:, None, :]
        cov_z = cov_z[:, None, :]

        # append state
        self.mu_z = torch.cat((self.mu_z, mu_z), dim=1)
        self.cov_z = torch.cat((self.cov_z, cov_z), dim=1)

    def reset(self, n_tsk):
        self.n_tsk = n_tsk

        # create intial state
        mu_z = torch.ones(self.d_z) * self.mu_z_init_scalar
        cov_z = torch.ones(self.d_z) * self.var_z_init_scalar

        # add task and states dimensions
        mu_z = mu_z[None, None, :]
        cov_z = cov_z[None, None, :]

        # expand task dimension
        mu_z = mu_z.expand(self.n_tsk, -1, -1)
        cov_z = cov_z.expand(self.n_tsk, -1, -1)

        # send to device and set attributes
        self.mu_z = mu_z.clone().to(self.device)
        self.mu_z_init = mu_z.clone().to(self.device)
        self.cov_z = cov_z.clone().to(self.device)
        self.cov_z_init = cov_z.clone().to(self.device)

    @staticmethod
    def _decode_step_input(latent_obs, agg_state_old):
        r = latent_obs[0]
        cov_r = latent_obs[1]
        mu_z = agg_state_old[0]
        cov_z = agg_state_old[1]
        return r, cov_r, mu_z, cov_z

    def agg2latent(self, agg_state):
        return agg_state[0], agg_state[1]

    def sequential_step(self, latent_obs, agg_state_old):
        # decode input
        r, cov_r, mu_z, cov_z = self._decode_step_input(
            latent_obs=latent_obs, agg_state_old=agg_state_old
        )

        # check input
        assert r.ndim == cov_r.ndim == 2
        assert r.shape == cov_r.shape
        assert mu_z.ndim == cov_z.ndim == 2
        assert mu_z.shape == cov_z.shape
        assert r.shape[0] == mu_z.shape[0]
        assert r.shape[1] == mu_z.shape[1]

        S = cov_z + cov_r
        S_inv = 1 / S
        K = cov_z * S_inv
        v = r - mu_z
        mu_z_new = mu_z + K * v
        cov_z_new = cov_z - K * cov_z

        return mu_z_new, cov_z_new

    def step(self, latent_obs, agg_state_old):
        # decode input
        r, cov_r, mu_z, cov_z = self._decode_step_input(
            latent_obs=latent_obs, agg_state_old=agg_state_old
        )

        # check input
        assert r.ndim == cov_r.ndim == 3
        assert r.shape == cov_r.shape
        assert mu_z.ndim == cov_z.ndim == 2
        assert mu_z.shape == cov_z.shape
        assert r.shape[0] == mu_z.shape[0]
        assert r.shape[2] == mu_z.shape[1]

        if r.shape[1] == 0:
            return mu_z, cov_z  # nothing to do

        if r.shape[1] == 1:
            # for one-point updates the sequential version is faster
            return self.sequential_step(
                latent_obs=(r[:, 0, :], cov_r[:, 0, :]), agg_state_old=(mu_z, cov_z)
            )

        v = r - mu_z[:, None, :]
        cov_w_inv = 1 / cov_r
        cov_z_new = 1 / (1 / cov_z + torch.sum(cov_w_inv, dim=1))
        mu_z_new = mu_z + cov_z_new * torch.sum(cov_w_inv * v, dim=1)

        return mu_z_new, cov_z_new

    def update_seq(self, latent_obs):
        # perform update
        new_agg_state = self.sequential_step(
            latent_obs=latent_obs, agg_state_old=self.last_agg_state
        )

        # append new state
        self._append_agg_state(agg_state=new_agg_state)

    def update(self, latent_obs):
        r = latent_obs[0]
        if r.shape[1] == 0:
            return  # nothing to do

        # perform update
        new_agg_state = self.step(
            latent_obs=latent_obs, agg_state_old=self.last_agg_state
        )

        # append new state
        self._append_agg_state(new_agg_state)


class MeanAggregator:
    def __init__(self, **kwargs):
        self.d_lo = kwargs["d_lo"]
        self.d_z = None
        self.mu_r_init_scalar = kwargs["init_data"][0]
        self.n_agg_init_scalar = kwargs["init_data"][1]

        self.mu_r_init = None
        self.n_agg_init = None
        self.mu_r = None  # the current mean
        self.n_agg = None  # the number of datapoints aggregated

        self.n_tsk = None  # number of tasks

        self.device = None

    @property
    def latent_state(self):
        return self.agg2latent(self.agg_state)

    @property
    def initial_latent_state(self):
        return self.agg2latent(self.initial_agg_state)

    @property
    def last_latent_state(self):
        return self.agg2latent(self.last_agg_state)

    @property
    def agg_state(self):
        return self.mu_r, self.n_agg

    @property
    def initial_agg_state(self):
        return self.mu_r_init[:, 0, :], self.n_agg_init[:, 0, :]

    @property
    def last_agg_state(self):
        return self.mu_r[:, -1, :], self.n_agg[:, -1, :]

    @property
    def parameters(self):
        return []

    def init_weights(self):
        pass

    def save_weights(self, logpath, epoch):
        pass

    def load_weights(self, logpath, epoch):
        pass

    def delete_all_weight_files(self, logpath):
        pass

    def set_device(self, device):
        self.device = device

    def _append_agg_state(self, agg_state):
        mu_r = agg_state[0]
        n_agg = agg_state[1]

        assert mu_r.ndim == 2

        # add ls dimension
        mu_r = mu_r[:, None, :]
        n_agg = n_agg[:, None, :]

        # append state
        self.mu_r = torch.cat((self.mu_r, mu_r), dim=1)
        self.n_agg = torch.cat((self.n_agg, n_agg), dim=1)

    def reset(self, n_tsk):
        self.n_tsk = n_tsk

        # create intial state
        mu_r = torch.ones(self.d_lo) * self.mu_r_init_scalar
        n_agg = torch.ones(1) * self.n_agg_init_scalar

        # add task and states dimensions
        mu_r = mu_r[None, None, :]
        n_agg = n_agg[None, None, :]

        # expand task dimension
        mu_r = mu_r.expand(self.n_tsk, -1, -1)
        n_agg = n_agg.expand(self.n_tsk, -1, -1)

        # send to device and set attributes
        self.mu_r = mu_r.clone().to(self.device)
        self.mu_r_init = mu_r.clone().to(self.device)
        self.n_agg = n_agg.clone().to(self.device)
        self.n_agg_init = n_agg.clone().to(self.device)

    def agg2latent(self, agg_state):
        return agg_state[0], None

    def _decode_step_input(self, latent_obs, agg_state_old):
        assert len(latent_obs) == 1
        r = latent_obs[0]
        n_r = torch.tensor(r.shape[1], dtype=torch.float).to(self.device)
        mu_r = agg_state_old[0]
        n_agg = agg_state_old[1]
        return r, n_r, mu_r, n_agg

    def step(self, latent_obs, agg_state_old):
        # decode input
        r, n_r, mu_r, n_agg = self._decode_step_input(
            latent_obs=latent_obs, agg_state_old=agg_state_old
        )

        # check input
        assert r.ndim == 3
        assert mu_r.ndim == 2
        assert r.shape[0] == mu_r.shape[0]
        assert r.shape[2] == mu_r.shape[1]
        assert r.shape[1] > 0

        mu_r_new = 1 / (n_r + n_agg) * (n_agg * mu_r + n_r * torch.mean(r, dim=1))
        n_agg_new = n_agg + n_r

        return mu_r_new, n_agg_new

    def update(self, latent_obs):
        r = latent_obs[0]
        if r.shape[1] == 0:
            return  # nothing to do

        # perform update
        new_agg_state = self.step(
            latent_obs=latent_obs, agg_state_old=self.last_agg_state
        )

        # append new state
        self._append_agg_state(new_agg_state)


class MeanAggregatorRtoZ(MeanAggregator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d_z = kwargs["d_z"]
        self.mean_z_mlp = MLP(
            name="agg2mean_z",
            d_in=kwargs["d_lo"],
            d_out=kwargs["d_z"],
            mlp_layers=kwargs["mlp_layers"],
            f_act=kwargs["f_act"],
            f_out=(None, {}),
            seed=kwargs["seed"],
        )
        self.cov_z_mlp = MLP(
            name="agg2cov_z",
            d_in=kwargs["d_lo"],
            d_out=kwargs["d_z"],
            mlp_layers=kwargs["mlp_layers"],
            f_act=kwargs["f_act"],
            f_out=("exp", {}),
            seed=kwargs["seed"],
        )

    def agg2latent(self, agg_state):
        return self.mean_z_mlp(agg_state[0]), self.cov_z_mlp(agg_state[0])

    def set_device(self, device):
        self.device = device
        self.mean_z_mlp.to(device)
        self.cov_z_mlp.to(device)

    @property
    def parameters(self):
        return list(self.mean_z_mlp.parameters()) + list(self.cov_z_mlp.parameters())

    def init_weights(self):
        pass

    def save_weights(self, logpath, epoch):
        self.mean_z_mlp.save(path=logpath, epoch=epoch)
        self.cov_z_mlp.save(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        self.mean_z_mlp.load_weights(path=logpath, epoch=epoch)
        self.cov_z_mlp.load_weights(path=logpath, epoch=epoch)

    def delete_all_weight_files(self, logpath):
        self.mean_z_mlp.delete_all_weight_files(path=logpath)
        self.cov_z_mlp.delete_all_weight_files(path=logpath)
