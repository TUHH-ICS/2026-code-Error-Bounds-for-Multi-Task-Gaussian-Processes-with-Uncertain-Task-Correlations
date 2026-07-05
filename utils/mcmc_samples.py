#!/usr/bin/env python3

"""
This module provides a utility function for obtaining MCMC samples from a Gaussian Process model using Pyro.
"""

import gpytorch
import pyro
from copy import deepcopy
from pyro.infer.mcmc import NUTS, MCMC
from torch import hstack, all, eye, vstack, einsum


def get_samples(gp, min_samples = 50, num_samples=100, warmup_steps=100):
    counter = 0
    num_tsks = gp.task_covar_module.covar_factor.size(-1)
    samples, diagnostics = run_mcmc(
        gp=gp, num_samples=num_samples, warmup_steps=warmup_steps
    )
    while samples["task_covar_module.covar_factor_prior"].shape[0] <= min_samples:
        if counter > 5:
            samples["task_covar_module.covar_factor_prior"] = (
                eye(num_tsks).unsqueeze(0).repeat(2, 1, 1)
            )
            break
        samples0, diagnostics = run_mcmc(
            gp=gp, num_samples=100, warmup_steps=100
        )
        samples = {
            key: vstack((samples[key], samples0[key]))
            for key in samples.keys()
        }
        counter += 1
    return samples, diagnostics


def run_mcmc(gp,num_samples=100, warmup_steps=100):
    train_inputs = gp.train_inputs[0]
    train_targets = gp.train_targets
    gppyro = deepcopy(gp)
    gppyro.task_covar_module.add_prior()
    gppyro.likelihood.noise = .1 #
    gppyro.train()

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled = gppyro.pyro_sample_from_prior()
            output = sampled.likelihood(sampled(x))
            pyro.sample("obs", output, obs=y.squeeze())
        return y

    nuts_kernel = NUTS(pyro_model, jit_compile=False, max_tree_depth=3, full_mass=False)
    mcmc_run = MCMC(
        nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=True
    )
    mcmc_run.run(train_inputs, train_targets)
    diagnostics = mcmc_run.diagnostics()
    samp_temp = mcmc_run.get_samples()
    C = samp_temp['task_covar_module.covar_factor_prior'] 
    row_0_cov = einsum('si,sji->sj', C[:, 0, :], C) 
    inds = (row_0_cov >= 0).all(dim=-1)
    samp_temp['task_covar_module.covar_factor_prior'] = C[inds]
    return samp_temp, diagnostics