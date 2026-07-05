#!/usr/bin/env python3

"""
This module provides functions for robust Gaussian Process (GP).
"""



import torch
from copy import deepcopy
from torch import Tensor
import torch
from math import floor
from matplotlib import pyplot as plt
from botorch.utils.sampling import draw_sobol_samples
from model.model import MultiTaskGPICM

def bayesian_robust_gp(
    sampmods,
    model0,
    bounds,
    delta: float = 0.05,
    tau: float = 0.01,
    rho = None,
    method: str = "standard",
    n_x: int = 100,
    n_mc: int = 10000,
):
    if rho is None:
        rho = delta
    robustmodel = deepcopy(model0)

    sqrtbeta = beta_bayes(bounds, tau, delta, method=method, n_x=n_x, n_mc=n_mc, gp=model0).sqrt()
    nu, sigprime = get_barbeta(
        model0, sampmods, sqrtbeta, rho
    )
    print(f"nu:{nu}")
    sqrtbetabar = sqrtbeta + nu
    print(f"sqrtbetabar: {sqrtbetabar}")
    sigmaprime = sigprime@sigprime.T
    print(f"Correlation Matrix: {sigmaprime}")
    if sqrtbeta <= sqrtbetabar*sigmaprime.det():
        robustmodel.task_covar_module._set_covar_factor(torch.eye(sigmaprime.size(0)))
        print("Using identity covariance matrix")
        return robustmodel, sqrtbeta
    else:
        robustmodel.task_covar_module._set_covar_factor(sigprime)
        print("Using correlation matrix")
        return robustmodel, sqrtbetabar


def beta_bayes(
    bounds: Tensor = torch.tensor([[0.0], [1.0]]),
    tau: float = 0.01,
    delta: float = 0.05,
    method: str = "standard",
    n_mc: int = 10000,
    n_x: int = 100,
    gp = None,
    batch_size: int = 1000,
):
    M = torch.hstack([torch.ceil((bu - bl) / (2*tau))  for bl, bu in bounds.T])
    m = M.prod()
    if method == "standard":
        beta = 2 * torch.log(m / delta)
    elif method == "MC":
        device = bounds.device
        
        x = draw_sobol_samples(bounds, n_x, q=1).reshape(n_x,-1).to(device=device)
        if isinstance(gp,MultiTaskGPICM):
            task = 0
            xt = torch.hstack((x, task * torch.ones(x.size(0), 1)))
        else:
            xt = x

        with torch.no_grad():
            posterior = gp.posterior(xt)
            mean = posterior.mean.squeeze(-1)                      # shape: n_x
            std = posterior.variance.sqrt().squeeze(-1).clamp_min(1e-12)

            T_list = []

            n_done = 0
            while n_done < n_mc:
                b = min(batch_size, n_mc - n_done)

                samples = posterior.rsample(torch.Size([b])).squeeze(-1)  # b x n_x

                Z = (samples - mean) / std
                T = Z.abs().amax(dim=-1)

                T_list.append(T)
                n_done += b

        T_all = torch.cat(T_list)

        beta = torch.quantile(T_all, 1 - delta)**2
    else:
        raise ValueError(f"Unknown method: {method}")
    return beta

def get_barbeta(model0, sampmods, maxsqrtbeta, rho: float):
    noise = model0.likelihood.noise.detach()
    sigmaf = model0.covar_module.outputscale.detach()
    covar = sampmods.task_covar_module._eval_covar_matrix()
    indmax = floor(covar.size(0) * (1 - rho))
    chol_covar = sampmods.task_covar_module.covar_factor.detach()
    dets = (torch.linalg.det(sigmaf*covar)+sigmaf*noise)/(sigmaf+noise)
    with torch.no_grad():
        nu = get_nu_optimized(sampmods)
    total = maxsqrtbeta+ nu
    total *= dets.sqrt().view(-1,1)
    total,inds = total.sort(dim=-1)
    target_totals = total[:, indmax]
    batch_indices = torch.arange(total.size(0), device=total.device)
    target_nus = nu[batch_indices, inds[:, indmax]]
    valid_mask = target_nus <= 10
    masked_totals = target_totals.clone()
    masked_totals[~valid_mask] = float('inf')
    if valid_mask.any():
        Id = masked_totals.argmin()
        sigprime = chol_covar[Id]
        nu = nu[Id,inds[Id,indmax]].detach()
    else:
        nu = 0
        sigprime = torch.eye(2)
    return nu, sigprime


def get_covar_factors(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix().detach()
    return torch.linalg.cholesky(covar, upper=False).squeeze()

def get_nu_optimized(sampmods):
    sampmods.train()
    
    train_inputs = sampmods.train_inputs[0] # Shape: (B, N, D)
    train_targets = sampmods.train_targets  # Shape: (B, N)
    
    train_tasks = train_inputs[0, :, -1].to(dtype=torch.int64) # Shape: (N,)
    noise = sampmods.likelihood.noise.detach()
    N = train_tasks.shape[0]
    
    K = sampmods(train_inputs).covariance_matrix # (B, N, N)
    k1 = sampmods.covar_module(train_inputs[:, :, :-1]).evaluate() # (B, N, N)
    
    Kd = K + noise * torch.eye(N, device=K.device).unsqueeze(0)
    alpha = torch.linalg.solve(Kd, train_targets.unsqueeze(-1)) # (B, N, 1)
    alpha_sq = alpha.squeeze(-1) # (B, N)
    
    # V_j = K_j @ alpha_j
    V = torch.matmul(K, alpha).squeeze(-1) # (B, N)
    
    # term1_i = alpha_i^T * K_i * alpha_i
    term1 = torch.einsum('bi, bi -> b', alpha_sq, V) # (B,)
    # term2_{i,j} = alpha_i^T * K_j * alpha_j = alpha_i^T * V_j
    term2 = torch.matmul(alpha_sq, V.T) # (B, B)
    
    covar_factors = sampmods.task_covar_module._eval_covar_matrix().detach() # (B, T, T)
    T_tasks = covar_factors.size(-1)
    
    P = torch.zeros(N, T_tasks, device=K.device, dtype=K.dtype)
    P.scatter_(1, train_tasks.unsqueeze(-1), 1.0)
    
    # Q_j = diag(alpha_j) @ P -> shape (B, N, T_tasks)
    Q = alpha_sq.unsqueeze(-1) * P.unsqueeze(0)
    
    # S_j = Q_j^T @ k1_j @ Q_j -> shape (B, T_tasks, T_tasks)
    S = torch.matmul(Q.transpose(-1, -2), torch.matmul(k1, Q))
    
    # M_j = C_j @ S_j @ C_j -> shape (B, T_tasks, T_tasks)
    C = covar_factors
    M = torch.matmul(C, torch.matmul(S, C))
    
    C_inv = torch.linalg.inv(C) # (B, T_tasks, T_tasks)
    
    # term3_{i,j} = Tr(C_i^-1 @ M_j)
    term3 = torch.einsum('iuv, jvu -> ij', C_inv, M) # (B, B)
    
    norm_diff = term1.unsqueeze(1) - 2 * term2 + term3 # (B, B)

    mean_se = post_mean_SE_optimized(sampmods)
    return norm_diff.add(mean_se).clamp(min=1e-12).sqrt()

def post_mean_SE_optimized(sampmods):
    noise = sampmods.likelihood.noise.detach()
    (train_inputs,) = sampmods.train_inputs
    sampmods.eval()
    
    # mu is shape (B, N)
    mu = sampmods(train_inputs).mean 
    
    # Optimization: ||mu_i - mu_j||^2 = ||mu_i||^2 - 2 * mu_i^T mu_j + ||mu_j||^2
    # This prevents creating a (B, B, N) tensor
    mu_sq = (mu ** 2).sum(dim=-1) # (B,)
    cross = torch.matmul(mu, mu.T) # (B, B)
    
    normdiff = (mu_sq.unsqueeze(1) - 2 * cross + mu_sq.unsqueeze(0)) / noise
    return normdiff

