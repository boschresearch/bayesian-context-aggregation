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

import numpy as np
import torch


class MySearchSpace:
    def __init__(self, bounds):
        self.bounds = bounds

    def get_continuous_bounds(self):
        return self.bounds


class GPBenchmark:
    def __init__(
        self, n_data_per_task, seed,
    ):
        self.rng_task = np.random.RandomState(seed)
        self.rng_data = np.random.RandomState(seed)
        self.n_data_per_task = n_data_per_task

        self.search_space = MySearchSpace(bounds=[[-2.0, 2.0]])

    def kernel(self, r):
        raise NotImplementedError

    def gram_matrix(self, x):
        distances = torch.nn.functional.pdist(x)
        gram_matrix_triu = self.kernel(distances)
        gram_matrix_diag = self.kernel(torch.tensor(0.0)) * torch.eye(x.shape[0])

        gram_matrix = torch.zeros((x.shape[0], x.shape[0]))
        triu_idx = np.triu_indices(x.shape[0], k=1)  # without diagonal
        gram_matrix[triu_idx[0], triu_idx[1]] = gram_matrix_triu
        gram_matrix = gram_matrix + gram_matrix.T + gram_matrix_diag

        return gram_matrix

    def generate_one_task(self):
        bounds = self.search_space.get_continuous_bounds()
        x = self.rng_data.uniform(
            low=bounds[0][0], high=bounds[0][1], size=(self.n_data_per_task, 1)
        )
        x = torch.tensor(x, dtype=torch.float)

        K = self.gram_matrix(x)
        # use double precision in cholesky for improved stability
        K = K.double()
        # add noise to diagonal to make cholesky stable
        K = K + 1e-5 * torch.eye(x.shape[0])
        cholesky = torch.cholesky(K)
        cholesky = cholesky.float()
        y = torch.matmul(
            cholesky,
            torch.tensor(self.rng_task.randn(*(x.shape[0], 1)), dtype=torch.float),
        )

        return x, y


class RBFGPBenchmark(GPBenchmark):
    def __init__(self, n_data_per_task, seed):
        super().__init__(n_data_per_task=n_data_per_task, seed=seed)

    def kernel(self, dist, lengthscale=1.0, signal_var=1.0):
        kernel_val = signal_var * torch.exp(-1 / 2 * dist ** 2 / lengthscale ** 2)
        return kernel_val


class Matern52GPBenchmark(GPBenchmark):
    def __init__(self, n_data_per_task, seed):
        super().__init__(n_data_per_task=n_data_per_task, seed=seed)

    def kernel(self, dist, lengthscale=0.25):
        kernel_val = (
            1 + np.sqrt(5) * dist / lengthscale + 5 * dist ** 2 / (3 * lengthscale ** 2)
        ) * torch.exp(-np.sqrt(5) * dist / lengthscale)
        return kernel_val


class WeaklyPeriodicGPBenchmark(GPBenchmark):
    def __init__(self, n_data_per_task, seed):
        super().__init__(n_data_per_task=n_data_per_task, seed=seed)

    def kernel(self, dist):
        kernel_val = torch.exp(-2 * torch.sin(1 / 2 * dist) ** 2 - 1 / 8 * dist ** 2)
        return kernel_val
