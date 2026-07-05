#!/usr/bin/env python3

"""
This module provides functions for multi-start and single-start optimization of Gaussian Processes (GPs) 
using PyTorch and GPyTorch."""

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from utils.utils import build_mtgp

max_trials = 10


def multistart_optimization(gp, num_restarts=1, mode=2, max_iter=20):
    (training_inputs,) = gp.train_inputs
    training_targets = gp.train_targets

    gp_vec = []
    loss_vec = torch.ones(num_restarts) * 10000
    for j in range(num_restarts):
        flag = False
        c = 0
        while not flag:
            if c == max_trials:
                raise TimeoutError("GP is not optimizable with mode {mode}...")
            gp = build_mtgp(
                (training_inputs[:, :-1], training_inputs[:, -1:]),
                training_targets.unsqueeze(-1),
            )
            gp, loss_vec[j], flag = optimize_gp(
                gp, mode=mode, max_iter=max_iter
            )
            c += 1
        gp_vec.append(gp)
    Id = torch.argmin(loss_vec)
    gp = gp_vec[Id]
    return gp


def optimize_gp(
    gp, mode=2, max_iter=20, train_covar=False, onlymt = False
):
    (training_inputs,) = gp.train_inputs
    training_targets = gp.train_targets

    # gp.mean_module.base_means[0].constant.requires_grad = False
    if not train_covar and hasattr(gp, "task_covar_module"):
        gp.task_covar_module.covar_factor.requires_grad = False

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp.train()
    losses = []
    if mode == 1:
        if not onlymt:
            optimizer = torch.optim.Adam([gp.task_covar_module.covar_factor], lr=0.1)
        else:
        # gp.task_covar_module.covar_factor.requires_grad = False
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
        for i in range(max_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(training_inputs)
            # Calc loss and backprop gradients
            loss = -mll(output, training_targets.squeeze())
            losses.append(loss.item())
            # if loss_vec[j] > loss.item(): loss_vec[j] = loss.item()
            # print(f"Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
    else:
        optimizer = torch.optim.LBFGS(
            gp.parameters(),
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            # tolerance_grad=1e-2,
        )

        def closure():
            optimizer.zero_grad()
            output = gp(training_inputs)
            loss = -mll(output, training_targets)
            losses.append(loss.item())
            # print(f"Loss: {loss.item():.4f}")
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except:
            Warning("Optimization failed")
            return None, 0, False
    # print(
    #     f"\nFinal loss: {losses[-1]}, Outputscale: {gp.covar_module.outputscale.item()}, Lengthscales: {gp.covar_module.base_kernel.lengthscale}, Means: {[i.constant.item() for i in gp.mean_module.base_means]} \n"
    # )
    return gp, losses[-1], True
