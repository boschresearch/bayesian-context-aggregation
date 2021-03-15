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

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm
from tqdm import tqdm


def compute_predictive_likelihood(model, dataloader, n_ctx_max):
    print(
        "Computing predictive log-likelihood over n_ctx_max = {:d}...".format(n_ctx_max)
    )

    # number of latent samples to evaluate sampling-based models
    if model.predictions_are_deterministic:
        n_marg = 1
    else:
        n_marg = 25

    # load test data
    x = []
    y = []
    for x_mb, y_mb in dataloader:
        x.append(x_mb.numpy())
        y.append(y_mb.numpy())
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    x_tst = x[:, n_ctx_max:, :]
    y_tst = y[:, n_ctx_max:, :]
    n_tsk = x_tst.shape[0]
    n_tst = x_tst.shape[1]
    d_y = y_tst.shape[2]

    # add marginalization dimension to y-tst
    y_tst = np.broadcast_to(y_tst[:, None, :, :], shape=(n_tsk, n_marg, n_tst, d_y))

    ll = 0
    for n_ctx in tqdm(range(n_ctx_max + 1)):
        # fit model
        x_ctx = x[:, :n_ctx, :]
        y_ctx = y[:, :n_ctx, :]
        model.fit(x=x_ctx, y=y_ctx)

        # obtain predictions
        mu_y, std_y = (
            np.zeros((n_tsk, n_marg, n_tst, d_y)),
            np.zeros((n_tsk, n_marg, n_tst, d_y)),
        )
        for i in range(n_marg):
            cur_mu_y, cur_var_y = model.predict(x_tst)
            cur_std_y = np.sqrt(cur_var_y)
            mu_y[:, i, :, :], std_y[:, i, :, :] = cur_mu_y, cur_std_y

        # compute log likelihood of data
        gaussian = norm(loc=mu_y, scale=std_y)
        cur_ll = gaussian.logpdf(y_tst)

        # sum log likelihood over output and datapoint dimensions
        cur_ll = np.sum(cur_ll, axis=(-2, -1))

        # compute MC-average
        cur_ll = scipy.special.logsumexp(cur_ll, axis=1)

        # sum task dimension
        cur_ll = np.sum(cur_ll, axis=0)
        assert cur_ll.ndim == 0

        # add -L * log(n_marg) with L = n_tsk
        cur_ll = -n_tsk * np.log(n_marg) + cur_ll

        # compute average log-likelihood over all datapoints
        cur_ll = cur_ll / (n_tsk * n_tst)

        ll += cur_ll

    ll /= n_ctx_max + 1

    return ll


def plot_prediction(model, dataloader, n_ctx, n_tsk):
    print("Plotting predictions...")

    # number of latent samples to evaluate sampling-based models
    if model.predictions_are_deterministic:
        n_marg = 1
    else:
        n_marg = 25

    # load test data
    x = []
    y = []
    for x_mb, y_mb in dataloader:
        x.append(x_mb.numpy())
        y.append(y_mb.numpy())
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    x_tst = x[:n_tsk, n_ctx:, :]
    y_tst = y[:n_tsk, n_ctx:, :]
    d_y = y_tst.shape[2]
    n_plt = 100
    x_plt = np.linspace(-2.0, 2.0, n_plt)[:, None]

    # fit model
    x_ctx = x[:n_tsk, :n_ctx, :]
    y_ctx = y[:n_tsk, :n_ctx, :]
    model.fit(x=x_ctx, y=y_ctx)

    # obtain predictions
    mu_y, std_y = (
        np.zeros((n_tsk, n_marg, n_plt, d_y)),
        np.zeros((n_tsk, n_marg, n_plt, d_y)),
    )
    for i in range(n_marg):
        cur_mu_y, cur_var_y = model.predict(x_plt)
        cur_std_y = np.sqrt(cur_var_y)
        mu_y[:, i, :, :], std_y[:, i, :, :] = cur_mu_y, cur_std_y
    mu_pred = np.mean(mu_y, axis=1)
    std_pred = np.std(mu_y, axis=1)
    output_noise_std_mean = np.mean(std_y, axis=1)
    std_pred += output_noise_std_mean

    # plot
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])
    for i in range(n_tsk):
        # draw new color for current task
        color = next(ax._get_lines.prop_cycler)["color"]

        # plot ground truth (=test set)
        sort_idx = np.argsort(x_tst[i].squeeze())
        ax.plot(
            x_tst[i, sort_idx].squeeze(),
            y_tst[i, sort_idx].squeeze(),
            color=color,
            alpha=0.5,
            lw=2,
            ls="--",
        )

        # plot context set
        ax.scatter(x_ctx[i].squeeze(), y_ctx[i].squeeze(), color=color)

        # plot predictions
        ax.plot(x_plt.squeeze(), mu_pred[i].squeeze(), color=color)
        ax.fill_between(
            x_plt.squeeze(),
            mu_pred[i].squeeze() - 2 * std_pred[i].squeeze(),
            mu_pred[i].squeeze() + 2 * std_pred[i].squeeze(),
            color=color,
            alpha=0.2,
        )
