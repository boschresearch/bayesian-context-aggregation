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

import numpy as np
import torch
import yaml
from tqdm import tqdm

from bayesian_aggregation.bayesian_aggregation.aggregator import (
    BayesianAggregator,
    MeanAggregator,
    MeanAggregatorRtoZ,
)
from bayesian_aggregation.bayesian_aggregation.decoder_network import (
    DecoderNetworkPB,
    DecoderNetworkSamples,
)
from bayesian_aggregation.bayesian_aggregation.encoder_network import (
    EncoderNetworkBA,
    EncoderNetworkMA,
)

encoder_dict = {
    "EncoderNetworkBA": EncoderNetworkBA,
    "EncoderNetworkMA": EncoderNetworkMA,
}
decoder_dict = {
    "DecoderNetworkPB": DecoderNetworkPB,
    "DecoderNetworkSamples": DecoderNetworkSamples,
}
aggregator_dict = {
    "BayesianAggregator": BayesianAggregator,
    "MeanAggregator": MeanAggregator,
    "MeanAggregatorRtoZ": MeanAggregatorRtoZ,
}


class CLVModel:
    def __init__(self, logpath):
        # paths, settings, normalizers, ...
        assert os.path.isdir(logpath)
        self.logpath = logpath
        self.f_settings = os.path.join(self.logpath, "000_settings.yaml")
        self.f_epoch = os.path.join(self.logpath, "000_epoch.yaml")
        self.f_normalizers = os.path.join(self.logpath, "000_normalizers.yaml")

        self.settings = None
        self.epoch = None
        self.normalizers = {
            "x_mu": None,
            "x_std": None,
            "y_mu": None,
            "y_std": None,
        }

        # model architecture
        self.encoder, self.decoder, self.aggregator = None, None, None
        self.device = None
        self.rng = None
        self.predictions_are_deterministic = None

        self.is_initialized = False

    def initialize_new_model(self, settings):
        # set settings and write them to file
        self.settings = settings
        self._write_settings_to_file()

        # set epoch and write it to file
        self.epoch = 0
        self._write_epoch_to_file()

        # initialize architecture
        self._create_encoder_decoder_aggregator()
        self._initialize_weights()
        self._set_device("cpu")
        self._set_deterministic_predictions_flag()

        # initialize random number generator
        self.rng = torch.Generator()
        self.rng.manual_seed(self.settings["training"]["seed"])

        self.is_initialized = True

    def load_model(self, load_epoch="last"):
        # load settings
        self.settings = self._load_settings_from_file()

        # set training epoch
        if load_epoch == "last":
            self.epoch = self._load_epoch_from_file()
        else:
            self.epoch = load_epoch
        assert self.epoch >= 0

        if self.epoch == 0:
            self.initialize_new_model(self.settings)
            return
        else:
            print("Loading model at epoch {:d}...".format(self.epoch))

        # load architecture
        self._create_encoder_decoder_aggregator()
        self._load_weights_from_file()
        self._load_normalizers_from_file()
        self._set_device("cpu")
        self._set_deterministic_predictions_flag()

        # initialize random number generator
        self.rng = torch.Generator()
        self.rng.manual_seed(self.settings["training"]["seed"])

        self.is_initialized = True

    def _write_settings_to_file(self):
        with open(self.f_settings, "w") as f:
            yaml.safe_dump(self.settings, f)

    def _load_settings_from_file(self):
        with open(self.f_settings, "r") as f:
            settings = yaml.safe_load(f)

        return settings

    def _write_epoch_to_file(self):
        with open(self.f_epoch, "w") as f:
            yaml.safe_dump(self.epoch, f)

    def _load_epoch_from_file(self):
        with open(self.f_epoch, "r") as f:
            epoch = yaml.safe_load(f)

        return epoch

    def _create_encoder_decoder_aggregator(self):
        self.encoder = encoder_dict[self.settings["encoder_type"]](
            **self.settings["encoder_kwargs"]
        )
        self.decoder = decoder_dict[self.settings["decoder_type"]](
            **self.settings["decoder_kwargs"]
        )
        self.aggregator = aggregator_dict[self.settings["aggregator_type"]](
            **self.settings["aggregator_kwargs"]
        )

    def _initialize_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.aggregator.init_weights()

    def _write_weights_to_file(self):
        self.encoder.save_weights(self.logpath, self.epoch)
        self.decoder.save_weights(self.logpath, self.epoch)
        self.aggregator.save_weights(self.logpath, self.epoch)
        self._write_epoch_to_file()

    def _load_weights_from_file(self):
        self.encoder.load_weights(self.logpath, self.epoch)
        self.decoder.load_weights(self.logpath, self.epoch)
        self.aggregator.load_weights(self.logpath, self.epoch)

    def _write_normalizers_to_file(self):
        # do this only once right at the beginning
        assert self.epoch == 0

        normalizers_as_lists = self.normalizers.copy()
        for (key, val) in normalizers_as_lists.items():
            normalizers_as_lists[key] = val.to("cpu").tolist()

        with open(self.f_normalizers, "w") as f:
            yaml.safe_dump(normalizers_as_lists, f)

    def _load_normalizers_from_file(self):
        with open(self.f_normalizers, "r") as f:
            self.normalizers = yaml.safe_load(f)
        for (key, val) in self.normalizers.items():
            self.normalizers[key] = torch.tensor(val)

    def _set_device(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.encoder.set_device(self.device)
        self.decoder.set_device(self.device)
        self.aggregator.set_device(self.device)

    def _set_deterministic_predictions_flag(self):
        self.predictions_are_deterministic = (
            self.settings["training"]["loss_type"] == "PB"
        )

    def meta_fit(self, dataloader_m, dataloader_v, max_epochs, tb_writer=None):
        print("Training model until epoch {:d}...".format(max_epochs))

        # define shorter names for some settings
        save_int = self.settings["training"]["save_interval"]
        val_int = self.settings["training"]["validation_interval"]
        adam_lr = self.settings["training"]["adam_lr"]

        # determine normalizers on metadata before starting training
        if self.epoch == 0:
            self._determine_normalizers(dataloader=dataloader_m)

        # set device for training
        self._set_device(self.settings["training"]["device"])

        # prepare optimizer
        optimizer = torch.optim.Adam(params=self._get_parameters(), lr=adam_lr)

        # training loop
        start_epoch = self.epoch + 1
        loss_v = None
        for self.epoch in tqdm(range(start_epoch, max_epochs + 1), mininterval=10):
            loss_m = 0.0

            # compute loss on minibatches of metadata
            for n_mb, (x_mb_m_nn, y_mb_m_nn) in enumerate(dataloader_m, 1):
                # reset gradient
                optimizer.zero_grad()

                # normalize data
                x_mb_m = self._normalize_x(x_mb_m_nn)
                y_mb_m = self._normalize_y(y_mb_m_nn)

                # send minibatch to device
                x_mb_m, y_mb_m = x_mb_m.to(self.device), y_mb_m.to(self.device)

                # sample context set size
                ctx_steps_m = [
                    torch.randint(
                        low=self.settings["training"]["n_context_meta_min"],
                        high=self.settings["training"]["n_context_meta_max"] + 1,
                        size=(1,),
                        generator=self.rng,
                    ).item()
                ]

                # compute loss
                loss_mb_m = self._compute_loss(
                    x=x_mb_m,
                    y=y_mb_m,
                    n_ctx=max(ctx_steps_m),
                    ctx_steps=ctx_steps_m,
                    loss_type=self.settings["training"]["loss_type"],
                )

                # perform gradient step on minibatch
                loss_mb_m.backward()
                optimizer.step()

                # sum up minibatch losses
                loss_m += loss_mb_m.item()

            # average losses over minibatches
            loss_m /= n_mb

            # log
            if tb_writer is not None:
                tb_writer.add_scalar("loss_meta", loss_m, self.epoch)

            # store weights
            if self.epoch % save_int == 0 or self.epoch == max_epochs:
                self._write_weights_to_file()

            # compute validation loss
            if (
                self.epoch == 1  # compute val loss at the beginning
                or self.epoch % val_int == 0  # ...and at each val_int
                or self.epoch == max_epochs  # ...and at the end
            ):
                with torch.no_grad():
                    loss_v = 0.0

                    for n_mb, (x_mb_v_nn, y_mb_v_nn) in enumerate(dataloader_v, 1):
                        # normalize data
                        x_mb_v = self._normalize_x(x_mb_v_nn)
                        y_mb_v = self._normalize_y(y_mb_v_nn)

                        # determine steps of context points at which to compute ll
                        ctx_steps_v = list(
                            range(
                                self.settings["training"]["n_context_val_min"],
                                self.settings["training"]["n_context_val_max"] + 1,
                            )
                        )

                        # send minibatch to device
                        x_mb_v, y_mb_v = x_mb_v.to(self.device), y_mb_v.to(self.device)

                        # compute loss ingredients
                        loss_mb_v = self._compute_loss(
                            x=x_mb_v,
                            y=y_mb_v,
                            n_ctx=max(ctx_steps_v),
                            ctx_steps=ctx_steps_v,
                            loss_type="PB"
                            if self.predictions_are_deterministic
                            else "MC",
                        )

                        # add up validation loss
                        loss_v += loss_mb_v.item()

                    # normalize validation loss w.r.t. minibatches
                    loss_v /= n_mb

                    # log
                    if tb_writer is not None:
                        tb_writer.add_scalar("loss_val", loss_v, self.epoch)

        self._set_device("cpu")

        # return validation loss
        return loss_v

    def _determine_normalizers(self, dataloader):
        # check that we've not already determined normalizers
        assert (self.normalizers[key] is None for key in self.normalizers.keys())

        # load full dataset (n_mb * batch_size tasks)
        x = []
        y = []
        for x_mb, y_mb in dataloader:
            x.append(x_mb)
            y.append(y_mb)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        # compute normalizers
        self.normalizers["x_mu"] = x.double().mean(dim=(0, 1)).float()
        self.normalizers["y_mu"] = y.double().mean(dim=(0, 1)).float()
        self.normalizers["x_std"] = x.double().std(dim=(0, 1)).float()
        self.normalizers["y_std"] = y.double().std(dim=(0, 1)).float()

        self._write_normalizers_to_file()

    def _normalize_x(self, x):
        normalizer_mu = self.normalizers["x_mu"][None, None, :]
        normalizer_std = self.normalizers["x_std"][None, None, :]
        x_normalized = (x - normalizer_mu) / normalizer_std

        return x_normalized

    def _normalize_y(self, y):
        normalizer_mu = self.normalizers["y_mu"][None, None, :]
        normalizer_std = self.normalizers["y_std"][None, None, :]
        y_normalized = (y - normalizer_mu) / normalizer_std

        return y_normalized

    def _denormalize_mu_y(self, mu_y):
        normalizer_mu = self.normalizers["y_mu"][None, None, :]
        normalizer_std = self.normalizers["y_std"][None, None, :]
        mu_y_denormalized = (mu_y * normalizer_std) + normalizer_mu

        return mu_y_denormalized

    def _denormalize_std_y(self, std_y):
        normalizer_std = self.normalizers["y_std"][None, None, :]
        std_y_denormalized = std_y * normalizer_std

        return std_y_denormalized

    def _get_parameters(self):
        return (
            self.encoder.parameters
            + self.decoder.parameters
            + self.aggregator.parameters
        )

    def _compute_loss(self, x, y, n_ctx, ctx_steps, loss_type):
        # create context and test sets
        x_ctx, y_ctx, x_tst, y_tst, latent_obs_tst = self._create_ctx_tst_sets(
            x=x, y=y, n_ctx=n_ctx, loss_type=loss_type
        )

        # reset aggregator
        self.aggregator.reset(x_ctx.shape[0])

        # encode all context points
        latent_obs_ctx = self.encoder.encode(x=x_ctx, y=y_ctx)

        # aggregate context data for all context set sizes
        pos = 0
        for i in range(len(ctx_steps)):
            cur_latent_obs = tuple(
                entry[:, pos : ctx_steps[i], :] for entry in latent_obs_ctx
            )
            self.aggregator.update(latent_obs=cur_latent_obs)
            pos = ctx_steps[i]

        # compute loss
        loss = 0.0
        ls = self.aggregator.latent_state
        agg_state = self.aggregator.agg_state
        mu_z = ls[0]
        cov_z = ls[1]
        if loss_type == "PB":
            loss -= self._conditional_log_likelihood(
                x=x_tst, y=y_tst, mu_z=mu_z, cov_z=cov_z
            )
        elif loss_type == "MC":
            loss -= self._marginal_log_likelihood(
                x=x_tst, y=y_tst, mu_z=mu_z, cov_z=cov_z
            )
        elif loss_type == "VI":
            loss -= self._vi_loss(
                x=x_tst,
                y=y_tst,
                mu_z=mu_z,
                cov_z=cov_z,
                agg_state=agg_state,
                latent_obs_tst=latent_obs_tst,
            )
        else:
            raise ValueError("Unknown loss loss_type!")

        return loss

    def _create_ctx_tst_sets(self, x, y, n_ctx, loss_type):
        n_all = x.shape[1]

        # determine context points
        idx_pts = torch.randperm(n=x.shape[1], generator=self.rng)
        x_ctx = x[:, idx_pts[:n_ctx], :]
        y_ctx = y[:, idx_pts[:n_ctx], :]

        if loss_type == "PB" or loss_type == "MC":
            # use remaining points as test points
            x_tst = x[:, idx_pts[n_ctx:], :]
            y_tst = y[:, idx_pts[n_ctx:], :]
            latent_obs_tst = None  # not necessary
        elif loss_type == "VI":
            # sample a test set from the remaining points
            n_tst = torch.randint(
                low=1, high=n_all - n_ctx, size=(1,), generator=self.rng
            ).squeeze()
            x_tst = x[:, idx_pts[n_ctx : n_ctx + n_tst], :]
            y_tst = y[:, idx_pts[n_ctx : n_ctx + n_tst], :]
            latent_obs_tst = self.encoder.encode(x_tst, y_tst)
        else:
            raise ValueError("Unknown loss loss_type!")

        return x_ctx, y_ctx, x_tst, y_tst, latent_obs_tst

    def _conditional_log_likelihood(self, x, y, mu_z, cov_z):
        assert x.ndim == y.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x.nelement() != 0
        assert y.nelement() != 0

        # obtain predictions
        mu_y, std_y = self._predict(x, mu_z, cov_z, n_marg=1)
        # mu_y, std_y shape = (n_tsk, n_ls, 1, n_tst, d_y)
        mu_y, std_y = mu_y.squeeze(2), std_y.squeeze(2)
        assert mu_y.ndim == 4 and std_y.ndim == 4

        # add latent state dimension to y-values
        n_ls = mu_y.shape[1]
        y = y[:, None, :, :].expand(-1, n_ls, -1, -1)

        # compute mean log-likelihood
        gaussian = torch.distributions.Normal(mu_y, std_y)
        ll = gaussian.log_prob(y)

        # take sum of lls over output dimension
        ll = torch.sum(ll, axis=-1)

        # take mean over all datapoints
        ll = torch.mean(ll)

        return ll

    def _marginal_log_likelihood(self, x, y, mu_z, cov_z):
        assert x.ndim == y.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x.nelement() != 0
        assert y.nelement() != 0

        # obtain predictions
        n_marg = self.settings["training"]["loss_kwargs"]["n_marg"]
        mu_y, std_y = self._predict(x, mu_z, cov_z, n_marg=n_marg)
        # mu_y, std_y shape = (n_tsk, n_ls, n_marg, n_tst, d_y)

        # add latent state and marginalization dimension to y-values
        n_ls = mu_y.shape[1]
        n_tsk = x.shape[0]
        n_tst = x.shape[1]
        assert n_marg > 0
        y = y[:, None, None, :, :].expand(-1, n_ls, n_marg, -1, -1)

        # compute log-likelihood for all datapoints
        gaussian = torch.distributions.Normal(mu_y, std_y)
        ll = gaussian.log_prob(y)

        # sum log-likelihood over output and datapoint dimension
        ll = torch.sum(ll, dim=(-2, -1))

        # compute MC-average
        ll = torch.logsumexp(ll, dim=2)

        # sum task- and ls-dimensions
        ll = torch.sum(ll, dim=(0, 1))
        assert ll.ndim == 0

        # add -L * log(n_marg) with L = n_tsk * n_ls
        ll = -n_tsk * n_ls * np.log(n_marg) + ll

        # compute average log-likelihood over all datapoints
        ll = ll / (n_tsk * n_ls * n_tst)

        return ll

    def _vi_loss(self, x, y, mu_z, cov_z, agg_state, latent_obs_tst):
        # computes the vi-inspired loss
        #  x, y may or may not include the context set
        #  latent_obs_tst are the latent observations w.r.t. the *test set only*
        #  mu_z, cov_z, agg_state are the latent/agg states w.r.t. the *context set*

        # obtain prior latent states
        prior_mu_z, prior_cov_z = mu_z, cov_z

        # obtain shapes
        n_ls = prior_mu_z.shape[1]
        n_tst = x.shape[1]

        # compute posterior latent states w.r.t. the test sets
        post_mu_zs = torch.zeros(prior_mu_z.shape, device=self.device)
        post_cov_zs = torch.zeros(prior_cov_z.shape, device=self.device)
        for j in range(n_ls):
            cur_agg_state_old = tuple(entry[:, j, :] for entry in agg_state)
            cur_agg_state_new = self.aggregator.step(
                agg_state_old=cur_agg_state_old, latent_obs=latent_obs_tst,
            )
            (cur_post_mu_z, cur_post_cov_z,) = self.aggregator.agg2latent(
                cur_agg_state_new
            )
            post_mu_zs[:, j, :] = cur_post_mu_z
            post_cov_zs[:, j, :] = cur_post_cov_z

        # compute log likelihood using posterior latent states
        ll = self._conditional_log_likelihood(
            x=x, y=y, mu_z=post_mu_zs, cov_z=post_cov_zs
        )

        # compute kls between posteriors and corresponding priors
        prior_std_z = torch.sqrt(prior_cov_z)
        post_std_zs = torch.sqrt(post_cov_zs)
        prior = torch.distributions.Normal(loc=prior_mu_z, scale=prior_std_z)
        posterior = torch.distributions.Normal(loc=post_mu_zs, scale=post_std_zs)
        kl = torch.distributions.kl.kl_divergence(posterior, prior)
        # sum over latent dimension (diagonal Gaussians)
        kl = torch.sum(kl, axis=-1)
        # take mean over task and ls dimensions
        kl = torch.mean(kl, dim=[0, 1]).squeeze()

        # compute loss
        elbo = ll - kl / n_tst

        return elbo

    def _predict(self, x, mu_z, cov_z, n_marg):
        assert x.ndim == 3  # (n_tsk, n_tst, d_x)
        assert mu_z.ndim == 3  # (n_tsk, n_ls, d_z)
        if self.settings["aggregator_type"] != "MeanAggregator":
            assert mu_z.shape == cov_z.shape

        # collect shapes
        n_tsk = mu_z.shape[0]
        n_ls = mu_z.shape[1]
        d_z = mu_z.shape[2]

        loss_type = self.settings["training"]["loss_type"]
        if loss_type == "PB":
            assert n_marg == 1
            if self.settings["aggregator_type"] != "MeanAggregator":
                mu_y, std_y = self.decoder.decode(x, mu_z, cov_z)
                # add dummy n_marg-dim
                mu_y, std_y = mu_y[:, :, None, :, :], std_y[:, :, None, :, :]
            else:
                # add dummy n_marg-dim
                mu_z = mu_z[:, :, None, :]
                mu_z = mu_z.expand(n_tsk, n_ls, n_marg, d_z)
                mu_y, std_y = self.decoder.decode(x, mu_z)
        elif loss_type == "VI" or loss_type == "MC":
            std_z = torch.sqrt(cov_z)

            # expand mu_z, std_z w.r.t. n_marg
            mu_z = mu_z[:, :, None, :]
            mu_z = mu_z.expand(n_tsk, n_ls, n_marg, d_z)
            std_z = std_z[:, :, None, :]
            std_z = std_z.expand(n_tsk, n_ls, n_marg, d_z)

            eps = torch.randn(size=mu_z.shape, generator=self.rng).to(self.device)
            z = mu_z + eps * std_z

            mu_y, std_y = self.decoder.decode(x, z)  # (n_tsk, n_ls, n_marg, n_tst, d_y)
        else:
            raise ValueError("Unknown loss loss_type!")

        assert mu_y.ndim == 5 and std_y.ndim == 5
        return mu_y, std_y  # (n_tsk, n_ls, n_marg, n_tst, d_y)

    @torch.no_grad()
    def predict(self, x):
        # prepare x
        has_tsk_dim = x.ndim == 3
        x = self._prepare_data_for_testing(x)
        x = self._normalize_x(x)

        # read out last latent state
        ls = self.aggregator.last_latent_state
        mu_z = ls[0][:, None, :]
        cov_z = ls[1][:, None, :] if ls[1] is not None else None

        # obtain predictions
        mu_y, std_y = self._predict(x, mu_z, cov_z, n_marg=1)
        mu_y, std_y = mu_y.squeeze(2), std_y.squeeze(2)  # squeeze n_marg dimension
        # pick prediction corresponding to last latent state
        mu_y, std_y = mu_y[:, -1, :, :], std_y[:, -1, :, :]

        # denormalize the predictions
        mu_y = self._denormalize_mu_y(mu_y)
        std_y = self._denormalize_std_y(std_y)

        # convert output back to caller structure
        if not has_tsk_dim:
            mu_y = mu_y.squeeze(0)
            std_y = std_y.squeeze(0)
        mu_y, std_y = mu_y.numpy(), std_y.numpy()

        return mu_y, std_y ** 2

    @torch.no_grad()
    def fit(self, x, y):
        # prepare x and y
        x = self._prepare_data_for_testing(x)
        y = self._prepare_data_for_testing(y)
        x = self._normalize_x(x)
        y = self._normalize_y(y)

        # accumulate data in aggregator
        self.aggregator.reset(n_tsk=x.shape[0])
        if x.shape[1] > 0:
            latent_obs = self.encoder.encode(x=x, y=y)
            self.aggregator.update(latent_obs)

    def _prepare_data_for_testing(self, data):
        data = torch.Tensor(data)
        if data.ndim == 1:
            data = data[None, None, :]  # add task and points dimension
        if data.ndim == 2:
            data = data[None, :, :]  # add task dimension
        data.to(self.device)

        return data
