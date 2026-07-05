#!/usr/bin/env python3

"""
This script reproduces the data used for the illustrations in the paper. It applies the SaMSBO algorithm on a specified function.

Usage:
    python3 run_samsbo.py <function_name> <disturbance_level>

Arguments:
    function_name (str): The name of the function to optimize. Must be one of:
        - MTBranin
        - LbSync
        - MTPowell
    disturbance_level (float): The level of disturbance to apply to the function.
"""


import os
import sys
import pickle
import random
import torch
import numpy.random
from math import ceil
from numpy import load
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

torch.set_default_dtype(torch.float64)

# Constants
NRUNS = 40
DELTA = 0.05
RHO = 0.15
TAU = 0.1

seeds = [729906250, 2355869572, 3026633807, 782243724, 2376833922, 3910350526, 299163792, 2403779625, 2079321245, 8259902, 3929387918, 409957175, 1745573805, 386744463, 2266843653, 3435227748, 1649667951, 103697523, 4001688108, 423900681, 1263232970, 3941478068, 3329194999, 2987314129, 2565035897, 2385613750, 1920821963, 250418623, 1234301914, 3688028183, 3841432629, 1845147206, 1124445604, 3268301151, 2953334578, 3339966651, 4108741113, 1065792445, 3316882374, 6300919]

def initialize_experiment(function_name, dist):
    """Initialize experiment parameters and directories."""
    folder = f"Final_runs_{int(100 * dist):03d}"
    os.makedirs(f"data/{folder}", exist_ok=True)

    data = load(f"data/X_init_{function_name.split('_')[0]}.npy", allow_pickle=True).item()
    return folder, torch.tensor(data["X_init"]), data["threshold"]

def configure_objective(function_name, dist, seed = None):
    """Configure the objective function and its parameters."""
    if seed is not None:
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

    if function_name == "LbSync":
        return LbSync(Ktyp="PI", num_lasers=5, num_tsks=2, disturbance=dist)
    else:
        raise ValueError(f"Unknown function name: {function_name}")

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
    robust_gp, sqrtbeta = bayesian_robust_gp(sample_models, gp, norm_bounds, delta=DELTA, tau=TAU, rho=RHO, method="standard", n_x=4096, n_mc=20000)
    return robust_gp, sqrtbeta

def main():
    function_name = sys.argv[1]
    dist = float(sys.argv[2])
    print(f"rho: {RHO}, delta: {DELTA} dist: {dist}")
    folder, X_init, T = initialize_experiment(function_name, dist)

    data_sets, bests, covars, betas = [], [], [], []

    for i in range(X_init.size(0)):
        torch.manual_seed(seeds[i])
        numpy.random.seed(seeds[i])
        random.seed(seeds[i])

        obj = configure_objective(function_name, dist)
        bounds, num_tsks = obj.bounds, obj.num_tsks

        print(f"Round: {i + 1}")
        x0 = X_init[i, ...].view(1, bounds.size(-1))
        norm_bounds = torch.vstack((torch.zeros(1, obj.dim), torch.ones(1, obj.dim)))

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

        train_inputs = unnormalize(norm_train_inputs, bounds)
        train_targets = bo.unstd_train_targets
        data_sets.append([train_inputs, train_tasks, train_targets])
        bests.append([bo.best_x, bo.best_y])
        covars.append(covar)
        betas.append(beta_)

        print(f"Best value: {round(bo.best_y[-1], 3)} at input: {unnormalize(bo.best_x[-1], bounds).round(decimals=3)}")

    with open(f"data/{folder}/{function_name}_dist_{int(100 * dist)}_new.obj", "wb") as file:
        pickle.dump({"data_sets": data_sets, "bests": bests, "covar": covars, "betas": betas, "T": T}, file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(len(sys.argv))
        print("Usage: python3 -m run_samsbo <function_name> <disturbance_level>") # LbSync, {0.05,0.15,0.25} in paper
        sys.exit(1)
    main()
