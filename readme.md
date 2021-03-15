# Bayesian Context Aggregation for Neural Processes
This is the source code accompanying the paper *Bayesian Context Aggregation for Neural Processes* by Volpp et al., ICLR 2021. The paper can be found [here](https://openreview.net/forum?id=ufZN2-aehFa). The code allows to reproduce results from the paper and to train neural processes with Bayesian context aggregation on new problems.

## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

## Installation
We kindly ask you to clone this repository and run

`pip install .`

from the root directory of this repository.

## Contents
We provide a script `bayesian_aggregation/scripts/train_evaluate_gps.py` to reproduce results from the GP-suite of experiments presented in the paper within error bounds.
This script trains the specified model for 200 epochs, computes the predictive likelihood, and plots predictions.
Weights are stored and re-used if this script is executed multiple times.

The script is called from the command line as follows:

`python train_evaluate_gps.py EXPERIMENT AGGREGATOR LIKELIHOOD_APPROXIMATION [-h]`

Description of the arguments:
1. `EXPERIMENT (str)`: the name of the experiment. Allowed values are:
`EXPERIMENT = {"RBFGP" | "WeaklyPeriodicGP" | "Matern52GP"}`
2. `AGGREGATOR (str)`: the aggregator to use. Allowed values are:
`AGGREGATOR = {"BA" | "MA"}`
3. `LIKELIHOOD_APPROXIMATION (str)`: the likelihood approximation to use. Allowed values are:
`LIKELIHOOD_APPROXIMATION = {"PB" | "VI" | "MC"}`

The combination "MA+PB" corresponds to the Conditional Neural Process (Garnelo et al., "Conditional Neural Processes", ICML 2018).

The combination "MA+VI" corresponds to the Neural Process (Garnelo et al., "Neural Processes", ICML 2018 Workshop on Theoretical Foundations and Applications of Deep Generative Models) without
a deterministic path.


## License 
"Bayesian Context Aggregation for Neural Processes" is open-sourced under the APGL-3.0 license. See the [LICENSE](LICENSE) file for details.

