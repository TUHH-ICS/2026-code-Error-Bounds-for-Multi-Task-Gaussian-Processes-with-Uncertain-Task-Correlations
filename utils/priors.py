#!/usr/bin/env python3

"""
This module contains custom prior distributions for use with Pyro and GPyTorch.
Classes:
    MultivariateNormalPriorPyro: A custom Multivariate Normal Prior compatible with Pyro.
    LKJCholeskyFactorPriorPyro: A custom LKJ Cholesky Factor Prior compatible with Pyro.
    LKJCovariancePriorPyro: A custom LKJ Covariance Prior compatible with Pyro.
    InverseWishartPriorPyro: A custom Inverse Wishart Prior compatible with Pyro.
"""



import torch
from gpytorch.priors import LKJCholeskyFactorPrior, MultivariateNormalPrior, LKJCovariancePrior
from gpytorch.priors.wishart_prior import InverseWishartPrior
from torch import Size
from torch._C import Size
from torch._tensor import Tensor

class MultivariateNormalPriorPyro(MultivariateNormalPrior):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=False, transform=None):
        super().__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args, transform)
    
    def expand(self,size: torch.Size, _instance = None):
        return self

class LKJCholeskyFactorPriorPyro(LKJCholeskyFactorPrior):
    def __init__(self, n, eta, validate_args=False, transform=None):
        super().__init__(n, eta, validate_args, transform)
    
    def expand(self, size: torch.Size, _instance = None):
        return self
    
class LKJCovariancePriorPyro(LKJCovariancePrior):
    def __init__(self, n, eta, sd_prior, validate_args=False):
        super().__init__(n, eta, sd_prior, validate_args)
    
    def expand(self, batch_shape, _instance=None):
        return self
    
class InverseWishartPriorPyro(InverseWishartPrior):
    def __init__(self, nu, K, validate_args=False):
        super().__init__(nu, K, validate_args)
        self.num_tasks = K.size(0)

    def expand(self, batch_shape: Size, _instance=None):
        return self
    
    def sample(self, sample_shape: Size = ...) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return torch.eye(self.num_tasks).expand(shape)