# %%
#!/usr/bin/env python3

"""
This script reproduces the data used for the illustrations in the paper. It applies the SaMSBO algorithm on a given function.

Usage:
    python3 -m run_bayes <function_name>

Arguments:
    function_name (str): The name of the function to optimize. Should be one of the following:
        - LbSync
"""

import torch
from math import ceil
from botorch.utils.transforms import normalize, unnormalize

from utils.utils import (
    sample_from_task,
    concat_data,
    standardize,
    build_mtgp,
)
from utils.mcmc_samples import get_samples
from utils.get_robust_gp import bayesian_robust_gp
from utils.optim import optimize_gp
from utils.functions import LbSync
from bo.bo_loop import MultiTaskBayesianOptimization
from utils.utils import plot_post
torch.set_default_dtype(torch.float64)

# Constants
NRUNS = 40
DELTA = 0.05
RHO = 0.15
TAU = 0.001
T=-15
DIST = 0.2

def initialize_experiment(obj, norm_bounds):
    """Initialize experiment parameters and directories."""
    d = obj.dim
    pot_start, _, pot_start_targets = sample_from_task(obj,[0], norm_bounds, n=128)
    ind = pot_start_targets[pot_start_targets.squeeze()>=0.9*T].squeeze().argmin()
    x0 = pot_start[pot_start_targets.squeeze()>=0.9*T][ind,...].view(1,d)
    return x0

def configure_objective(dist):
    """Configure the objective function and its parameters."""

    return LbSync(Ktyp="P", num_lasers=1, num_tsks=2, disturbance=dist)

def evaluate_initial_points(obj, x0, bounds):
    """Evaluate initial points for all tasks."""
    norm_x0 = normalize(x0, bounds)
    train_targets = torch.zeros(obj.num_tsks, 1)
    for j in range(obj.num_tsks):
        train_targets[j, ...] = obj.f(norm_x0, j)
    return norm_x0, train_targets

def evaluate_supplementary_tasks(obj, norm_bounds, norm_train_inputs, train_tasks, train_targets):
    """Evaluate supplementary tasks and concatenate data."""
    for k in range(1, obj.num_tsks):
        x, t, y = sample_from_task(obj, [k], norm_bounds, n=2 * ceil(2 * obj.dim / (obj.num_tsks - 1)))
        norm_train_inputs, train_tasks, train_targets = concat_data((x, t, y), (norm_train_inputs, train_tasks, train_targets))
    return norm_train_inputs, train_tasks, train_targets

def optimize_and_update_gp(gp, num_tsks, norm_bounds):
    """Optimize and update the Gaussian Process model."""
    gp, _, _ = optimize_gp(gp, mode=1, max_iter=200)
    mu = [gp.mean_module.base_means[r].constant.detach() for r in range(num_tsks)]
    samples, _ = get_samples(gp, min_samples=50, num_samples=100, warmup_steps=100)
    norm_train_inputs, train_tasks = gp.train_inputs[0][:,:-1], gp.train_inputs[0][:, -1:].to(dtype=torch.int32)
    norm_train_targets = gp.train_targets.unsqueeze(-1)
    sample_models = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets)
    sample_models.task_covar_module.add_prior()
    sample_models.pyro_load_from_samples(samples)
    gp = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets, mu=mu)
    robust_gp, sqrtbeta = bayesian_robust_gp(sample_models, gp, norm_bounds, delta=DELTA, tau=TAU, rho=RHO)
    return robust_gp, sqrtbeta

def main():
    torch.manual_seed(0)
    print(f"rhomax: {RHO}, dist: {DIST}")
    obj = configure_objective(DIST)
    bounds, num_tsks = obj.bounds, obj.num_tsks

    norm_bounds = torch.vstack((torch.zeros(1, obj.dim), torch.ones(1, obj.dim)))
    x0 = unnormalize(initialize_experiment(obj, norm_bounds),bounds)
    norm_x0, train_targets = evaluate_initial_points(obj, x0, bounds)
    train_tasks = torch.arange(num_tsks).unsqueeze(-1)
    norm_train_inputs = norm_x0.repeat(num_tsks, 1)

    norm_train_inputs, train_tasks, train_targets = evaluate_supplementary_tasks(
        obj, norm_bounds, norm_train_inputs, train_tasks, train_targets
    )

    norm_train_targets = standardize(train_targets, T=T)
    T_stdizd = standardize(T, T)
    num_acq_samps = [1] + [ceil(2 * obj.dim / (num_tsks - 1))] * (num_tsks - 1)
    bo = MultiTaskBayesianOptimization(obj, list(range(num_tsks)), norm_bounds, T_stdizd, T, num_acq_samps)
    covar = torch.zeros(NRUNS, num_tsks, num_tsks)
    beta_ = torch.zeros(NRUNS)
    mod_runs = 4
    sqrtbeta = 1.

    for run in range(NRUNS):
        gp = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets)

        if run >= 45:
            bo.num_acq_samps = [1] * num_tsks
            mod_runs = 15
        if run <= 10 or run % mod_runs == 0:
            robust_gp, sqrtbeta = optimize_and_update_gp(
                gp, num_tsks, norm_bounds
            )
        else:
            gp, _, _ = optimize_gp(gp, mode=1, max_iter=200)
            mu = [gp.mean_module.base_means[r].constant.detach() for r in range(num_tsks)]
            covar_chol = robust_gp.task_covar_module.covar_factor
            robust_gp = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets, mu=mu)
            robust_gp.task_covar_module._set_covar_factor(covar_chol)
        covar[run, ...] = robust_gp.task_covar_module._eval_covar_matrix()
        beta_[run] = sqrtbeta
        bo.update_gp(robust_gp, sqrtbeta)
        norm_train_inputs, train_tasks, norm_train_targets = bo.step()
        plot_post(robust_gp,[0],torch.linspace(0,1,100),sqrtbeta,T_stdizd)

    print(f"Best value: {round(bo.best_y[-1], 3)} at input: {unnormalize(bo.best_x[-1], bounds).round(decimals=3)}")

if __name__ == "__main__":
    main()



