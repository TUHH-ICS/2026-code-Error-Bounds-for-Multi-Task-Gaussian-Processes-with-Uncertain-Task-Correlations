#/usr/bin/python3

"""
This module provides utility functions for building and evaluating multi-task Gaussian Process (GP) models using GPyTorch.
"""


import torch
from torch import Tensor
from botorch.utils import draw_sobol_samples
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from model.model import MultiTaskGPICM
from cov.task_cov import IndexKernelAllPriors
from gpytorch.means import MultitaskMean
from gpytorch.means import ConstantMean
from utils.priors import LKJCholeskyFactorPriorPyro
from copy import deepcopy
from gpytorch.constraints import GreaterThan
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel


# Build multi-task GP model
def build_mtgp(
    train_inputs,
    train_targets,
    varf=.3,
    mu=None,
    likelihood=None,
    kernel=None,
):
    # Set default likelihood if not provided
    if likelihood is None:
        likelihood = GaussianLikelihood()
        likelihood.noise = torch.tensor(1e-4)

    # Determine the number of tasks
    num_tasks = train_inputs[-1].to(dtype=torch.int64).max() + 1

    # Set default mean values if not provided
    if mu is None:
        mu = torch.zeros(num_tasks)

    # Prepare training inputs
    d = train_inputs[0].shape[-1]
    train_inputs = torch.hstack(train_inputs).squeeze()

    # Initialize the multi-task GP model
    gp = MultiTaskGPICM(
        train_inputs,
        train_targets,
        task_feature=-1,
        likelihood=likelihood,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)),
        task_covar_module=IndexKernelAllPriors(
            num_tasks=num_tasks,
            rank=num_tasks,
            covar_factor_prior=LKJCholeskyFactorPriorPyro(num_tasks, .1),
        ),
        mean_module=MultitaskMean([ConstantMean()], num_tasks),
    )

    # Set task covariance matrix
    chol_covar = torch.linalg.cholesky(torch.tensor([[1.0, 0.95], [0.95, 1.0]]))
    gp.task_covar_module._set_covar_factor(chol_covar)

    # Set output scale for the covariance module
    gp.covar_module._set_outputscale(varf)


    # Set kernel lengthscale if not provided
    if kernel is None:
        ell = 0.08
        gp.covar_module.base_kernel._set_lengthscale(
            torch.tensor([ell]).repeat(1, int(d))
        )

    # Set mean values for each task
    for i in range(num_tasks):
        gp.mean_module.base_means[i]._constant_closure(
            gp.mean_module.base_means[i], mu[i]
        )

    return gp

    # Build single-task GP model
def build_stgp(
    train_inputs,
    train_targets,
    varf=.3,
    mu=None,
    likelihood=None,
    kernel=None,
):

    # Set default likelihood if not provided
    if likelihood is None:
        likelihood = GaussianLikelihood()
        likelihood.noise = torch.tensor(1e-4)

    # Set default mean value if not provided
    if mu is None:
        mu = torch.tensor(0.0)

    # Prepare training inputs
    d = train_inputs.shape[-1]

    # Initialize the single-task GP model
    gp = SingleTaskGP(
        train_X=train_inputs,
        train_Y=train_targets,
        likelihood=likelihood,
        outcome_transform=None,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)),
        mean_module=ConstantMean(),
    )

    # Set output scale for the covariance module
    gp.covar_module._set_outputscale(varf)

    # Set kernel lengthscale if not provided
    if kernel is None:
        ell = 0.08
        gp.covar_module.base_kernel._set_lengthscale(
            torch.tensor([ell]).repeat(1, int(d))
        )

    # Set mean value
    gp.mean_module.initialize(constant=mu)

    return gp


# Plot posterior of one-dimensional problems
def plot_post(gp, task, test_x, sqrtbeta=None, threshold=None):
    figure = plt.figure(1)
    figure.clf()
    with torch.no_grad():
        posterior = gp.posterior(test_x.reshape( -1, 1, 1), output_indices=task)
        ymean = posterior.mean.squeeze()
        if sqrtbeta != None:
            std_dev = posterior.variance.squeeze().sqrt()
            u = ymean + sqrtbeta * std_dev
            l = ymean - sqrtbeta * std_dev
        else:
            l, u = posterior.mvn.confidence_region()
            u = u.detach().numpy(); l = l.detach().numpy()
    ax = figure.add_subplot(1, 1, 1)
    x = gp.train_inputs[0]
    i = x[:, -1]
    x = x[:, 0:-1]
    i = i.squeeze()
    y = gp.train_targets
    ax.fill_between(test_x, u.squeeze(), l.squeeze(), alpha=0.2, color="C0")
    ax.plot(test_x, ymean, "C0")
    ax.plot(x[i == 1], y[i == 1], "xC1")
    x1 = x[i == 0]; y1 = y[i == 0]
    ax.plot(x1[:-1], y1[:-1], "xC2")
    ax.plot(x1[-1],y1[-1],'oC2')
    if threshold is not None:
        ax.plot(test_x,threshold*torch.ones_like(test_x),'--')
    plt.show()
    

# Standardize and unstandardize data with respect to primary task
# def standardize(train_y, mu = None, std = None, train_task = None):
#     if isinstance(train_y,Tensor):
#         train_y = deepcopy(train_y.detach())
#     if mu == None or std == None:
#         if train_task != None:
#             train_task = train_task.squeeze().to(dtype=torch.int32)
#             num_tasks = max(train_task)+1
#             mu = torch.hstack([torch.mean(train_y[train_task==i]) for i in range(num_tasks)])
#             norm_train_y = torch.cat([train_y[train_task==i]-mu[i] for i in range(num_tasks)])
#             std = torch.std(norm_train_y).nan_to_num(1.0).view(-1,1)
#             train_y = (train_y-mu[0])/std[0]
#             print(f"Std: {std}"); print(f"Mean: {mu}")
#             return train_y, mu, std 
#         else:
#             std,mu = torch.std_mean(train_y)
#             print(f"Std: {std.item()}")
#             print(f"Mean: {mu.item()}")
#             return (train_y-mu)/std,mu.view(-1,1),std.view(-1,1)
#     norm_train_y = (train_y-mu[0])/std[0]
#     return norm_train_y, mu, std

# def unstandardize(train_y, mu, std, train_task = None):
#     if train_task != None and mu.numel() > 1:
#         train_task=train_task.squeeze().to(dtype=torch.int32)
#         train_y = deepcopy(train_y)
#         num_tasks = train_task.max()+1
#         for i in range(num_tasks):
#             train_y[train_task==i] = train_y[train_task==i]*std[0] + mu[0]
#         return train_y
#     else:
#         train_y = train_y*std[0] + mu[0]
#         return train_y

def standardize(train_y, T, factor = 3.0):
    if isinstance(train_y,Tensor):
        train_y = deepcopy(train_y.detach())
    std = abs(T)/factor
    return train_y/std
     

def unstandardize(train_y, T, factor = 3.0):
    std = abs(T)/factor
    return train_y*std


# Seperate data by task
def seperate_data(train_inputs,train_targets,train_task=None):
    if train_task != None:
        train_inputs = torch.hstack((train_inputs,train_task))
    train_x, train_t = train_inputs[:,:-1], train_inputs[:,-1:]
    train_t = train_t.squeeze().to(dtype=torch.int32)
    num_tasks = train_t.max().item()+1
    d = train_x.shape[-1]
    sep_train_x = [train_x[train_t==i].view(-1,d) for i in range(num_tasks)]
    sep_train_y = [train_targets[train_t.squeeze()==i].view(-1,1) for i in range(num_tasks)]
    return sep_train_x,sep_train_y


# Sort data by task
def sort_data(train_inputs,train_targets):
    for i in range(len(train_inputs)):
        train_inputs[i],indices = train_inputs[i].sort()
        train_inputs[i] = train_inputs[i][indices]
        train_targets[i] = train_targets[i][indices]
    return train_inputs,train_targets


# Sample data from tasks
def sample_from_task(obj, tasks, bounds, n=5, data = None):
    for i in tasks:
        ni = n[i] if isinstance(n,list) else n
        X_init = draw_sobol_samples(bounds = bounds, n=ni, q=1).reshape(ni,bounds.size(-1))
        data = concat_data((X_init,
                    i*torch.ones(X_init.size(0),1),
                    obj.f(X_init,i)),data)
    return data

# Concatenate data
def concat_data(data, mem = None):
    if mem is None:
        return data
    else:
       return tuple([torch.vstack((mem,data)) for mem,data in zip(mem,data)])
