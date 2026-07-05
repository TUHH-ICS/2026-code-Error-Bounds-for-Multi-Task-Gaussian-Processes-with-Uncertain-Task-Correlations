#!/usr/bin/env python3

"""
Script to run different Bayesian Optimization algorithms:
    - Single-task with constraints
    - Single-task without constraints
    - Multi-task without constraints

Usage:
    python3 -m run_bayes <function_name> <algorithm_type>

Arguments:
    function_name (str): The name of the function to optimize. Options:
        - LbSync
    algorithm_type (str): The type of algorithm to run. Options:
        - "st_constraints" (Single-task with constraints)
        - "st_no_constraints" (Single-task without constraints)
        - "mt_no_constraints" (Multi-task without constraints)
"""

import torch
from botorch.utils.transforms import normalize, unnormalize
from utils.utils import sample_from_task, concat_data, standardize, build_mtgp, build_stgp
from utils.get_robust_gp import beta_bayes
from utils.optim import optimize_gp
from utils.functions import LbSync
from bo.bo_loop import SingleTaskBayesianOptimization, MultiTaskBayesianOptimization
from numpy import load
from math import ceil
import sys, os, pickle, random, numpy

torch.set_default_dtype(torch.float64)

# Global constants
NRUNS = 50
DELTA = 0.05
TAU = 0.1
DIST = 0.1


def initialize_function(function_name):
    """Initialize the objective function and related parameters."""
    if function_name == "LbSync":
        obj = LbSync(Ktyp="PI", num_lasers=5, num_tsks=2, disturbance=DIST)
    else:
        raise ValueError(f"Unknown function name: {function_name}")
    return obj, obj.dim, obj.bounds


def load_initial_data(function_name):
    """Load initial data points and thresholds."""
    data = load(f"data/X_init_{function_name}.npy", allow_pickle=True).item()
    return torch.tensor(data["X_init"]), data["threshold"]


def setup_directories(folder):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(f"data/{folder}"):
        os.mkdir(f"data/{folder}")


def run_single_task_with_constraints(function_name, X_init, T):
    """Run single-task Bayesian Optimization with constraints."""
    folder = "Bayes_ST"
    setup_directories(folder)
    return run_bayesian_optimization(function_name, X_init, T, folder, single_task=True, constraints=True)


def run_single_task_without_constraints(function_name, X_init, T):
    """Run single-task Bayesian Optimization without constraints."""
    folder = "vanilla_bo"
    setup_directories(folder)
    return run_bayesian_optimization(function_name, X_init, T, folder, single_task=True, constraints=False)


def run_multi_task_without_constraints(function_name, X_init, T):
    """Run multi-task Bayesian Optimization without constraints."""
    folder = "Bayes_MT"
    setup_directories(folder)
    return run_bayesian_optimization(function_name, X_init, T, folder, single_task=False, constraints=True)

def build_gp(train_inputs, train_task, train_targets, single_task=True):
    """Build a Gaussian Process model."""
    if single_task:
        return build_stgp(train_inputs, train_targets)
    else:
        return build_mtgp((train_inputs,train_task), train_targets)
        
def run_bayesian_optimization(function_name, X_init, T, folder, single_task, constraints):
    """Generalized Bayesian Optimization loop."""

    data_sets = []
    bests = []

    for i in range(X_init.size(0)):
        torch.manual_seed(seeds[i])
        numpy.random.seed(seeds[i])
        random.seed(seeds[i])


        obj, d, bounds = initialize_function(function_name)
        num_tsks = 1 if single_task else obj.num_tsks
        norm_bounds = torch.vstack((torch.zeros(1, d), torch.ones(1, d)))
        num_sup_task_samples = 1 if single_task else ceil(2 * d / (num_tsks - 1))
        num_acq_samps = [1] + [num_sup_task_samples] * (num_tsks - 1)
        print(f"Round: {i + 1}")

        x0 = X_init[i, ...].view(1, bounds.size(-1))
        norm_x0 = normalize(x0, bounds)

        # Evaluate initial point for all tasks
        train_targets, train_tasks, norm_train_inputs = initialize_training_data(obj, norm_x0, num_tsks)

        # Evaluate supplementary tasks if multi-task
        if not single_task:
            norm_train_inputs, train_tasks, train_targets = evaluate_supplementary_tasks(
                obj, num_tsks, norm_bounds, num_sup_task_samples, norm_train_inputs, train_tasks, train_targets
            )

        norm_train_targets = standardize(train_targets, T=T)
        T_stdizd = standardize(T, T)
        if single_task:
            bo = SingleTaskBayesianOptimization(
            obj, norm_bounds, T_stdizd, T, num_acq_samps, constraints=constraints
            )
        else:
            bo = MultiTaskBayesianOptimization(
            obj, list(range(num_tsks)), norm_bounds, T_stdizd, T, num_acq_samps, constraints=constraints
            )
        gp = build_gp(norm_train_inputs, train_tasks, norm_train_targets, single_task = single_task)

        bo = run_bo_iterations(bo, gp, norm_bounds, norm_train_inputs, train_tasks, norm_train_targets, single_task, constraints)

        train_inputs = unnormalize(bo.train_inputs, bounds)
        train_targets = bo.unstd_train_targets
        data_sets.append([train_inputs, torch.zeros(1,1) if single_task else bo.train_tasks, train_targets])
        bests.append([bo.best_x, bo.best_y])
        print(f"Best value: {round(bo.best_y[-1], 3)} at input: {unnormalize(bo.best_x[-1], bounds).round(decimals=3)}")

    # Save data
    save_results(folder, data_sets, bests)


def initialize_training_data(obj, norm_x0, num_tsks):
    """Initialize training data with the initial point."""
    train_targets = torch.zeros(num_tsks, 1)
    for j in range(num_tsks):
        train_targets[j, ...] = obj.f(norm_x0, j)
    train_tasks = torch.arange(num_tsks).unsqueeze(-1)
    norm_train_inputs = norm_x0.repeat(num_tsks, 1)
    return train_targets, train_tasks, norm_train_inputs


def evaluate_supplementary_tasks(obj, num_tsks, norm_bounds, num_sup_task_samples, norm_train_inputs, train_tasks, train_targets):
    """Evaluate supplementary tasks for multi-task optimization."""
    for k in range(1, num_tsks):
        x, t, y = sample_from_task(obj, [k], norm_bounds, n=2 * num_sup_task_samples)
        norm_train_inputs, train_tasks, train_targets = concat_data(
            (x, t, y), (norm_train_inputs, train_tasks, train_targets)
        )
    return norm_train_inputs, train_tasks, train_targets


def run_bo_iterations(bo, gp, norm_bounds, norm_train_inputs, train_tasks, norm_train_targets, single_task, constraints):
    """Run the Bayesian Optimization iterations."""
    for _ in range(NRUNS):
        if constraints or _ % 2 == 0:
            robust_gp = build_gp(norm_train_inputs, train_tasks, norm_train_targets, single_task=single_task)
            # robust_gp, _, _ = optimize_gp(robust_gp, mode=1, max_iter=200, train_covar=not single_task, onlymt=True)
            # covar_fac = robust_gp.task_covar_module.covar_factor.detach()
            # print(covar_fac@covar_fac.T)
        else:
            gp = build_gp(norm_train_inputs, train_tasks, norm_train_targets, single_task=single_task)
            gp, _, _ = optimize_gp(gp, mode=1, max_iter=200, train_covar=not single_task)
            robust_gp = gp
        sqrtbeta = beta_bayes(norm_bounds, TAU, DELTA, method="MC", n_x=4096, n_mc=20000, gp=robust_gp).sqrt()
        print(f"sqrtbeta: {sqrtbeta.item()}")
        bo.update_gp(robust_gp, sqrtbeta)
        norm_train_inputs, train_tasks, norm_train_targets = bo.step()
    return bo


def save_results(folder, data_sets, bests):
    """Save the results to a file."""
    file = open(f"data/{folder}/{function_name}_MC.obj", "wb")
    pickle.dump({"data_sets": data_sets, "bests": bests}, file)
    file.close()


# Define a dictionary of seeds for each function_name
seeds = [729906250, 2355869572, 3026633807, 782243724, 2376833922, 3910350526, 299163792, 2403779625, 2079321245, 8259902, 3929387918, 409957175, 1745573805, 386744463, 2266843653, 3435227748, 1649667951, 103697523, 4001688108, 423900681, 1263232970, 3941478068, 3329194999, 2987314129, 2565035897, 2385613750, 1920821963, 250418623, 1234301914, 3688028183, 3841432629, 1845147206, 1124445604, 3268301151, 2953334578, 3339966651, 4108741113, 1065792445, 3316882374, 6300919]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 -m run_comparison <function_name> <algorithm_type>")
        sys.exit(1)

    function_name = sys.argv[1]
    algorithm_type = sys.argv[2]

    print(algorithm_type)

    X_init, T = load_initial_data(function_name)
    print(X_init.size())

    if algorithm_type == "st_constraints":
        run_single_task_with_constraints(function_name, X_init, T)
    elif algorithm_type == "st_no_constraints":
        run_single_task_without_constraints(function_name, X_init, T)
    elif algorithm_type == "mt_no_constraints":
        run_multi_task_without_constraints(function_name, X_init, T)
    else:
        print(f"Unknown algorithm type: {algorithm_type}")
        sys.exit(1)

