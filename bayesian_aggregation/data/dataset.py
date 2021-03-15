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
from torch.utils.data import IterableDataset

from bayesian_aggregation.bayesian_aggregation.util import benchmark_dict


class NPIterableDataset(IterableDataset):
    def __init__(self, benchmark_name, n_task, n_data_per_task, seed):
        super(NPIterableDataset).__init__()
        self.benchmark = benchmark_dict[benchmark_name](
            n_data_per_task=n_data_per_task, seed=seed
        )
        self.n_task = n_task

    def __iter__(self):
        for _ in range(self.n_task):
            x, y = self.benchmark.generate_one_task()
            yield torch.Tensor(x), torch.Tensor(y)
