#!/usr/bin/env python3

"""
This module defines the `IndexKernelAllPriors` class, which extends the `IndexKernel` class from GPyTorch (https://docs.gpytorch.ai/en/stable/kernels.html#specialty-kernels) to include additional priors and constraints.

Classes:
    IndexKernelAllPriors: A kernel that allows for the inclusion of priors and constraints on the covariance factor and variance.
"""


import torch
from torch import Tensor
from gpytorch.priors import Prior
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels import Kernel



class IndexKernelAllPriors(IndexKernel):
    def __init__(
        self,
        num_tasks: int,
        rank: int,
        const: float = 0.01,
        covar_factor_prior: Prior | None = None,
        covar_factor_constraint: None = None,
        **kwargs
    ):
        super(IndexKernelAllPriors, self).__init__(
            num_tasks, rank, None, None, **kwargs
        )
        self.rank = rank
        self.num_tasks = num_tasks
        self.const = torch.eye(self.num_tasks)*const
        if covar_factor_prior is not None:
            if not isinstance(covar_factor_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(covar_factor_prior).__name__
                )
            self.covar_factor_prior = covar_factor_prior

        if covar_factor_constraint is not None:
            self.register_constraint("raw_covar_factor", covar_factor_constraint)

    def _set_covar_factor(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.covar_factor)

        self.initialize(
            covar_factor=self.covar_factor_constraint.inverse_transform(value)
            if hasattr(self, "covar_factor_constraint")
            else value
        )

    def _covar_factor_param(self, m: Kernel) -> Tensor:
        return m.covar_factor

    def _covar_factor_closure(self, m: Kernel, v: Tensor) -> Tensor:
        return m._set_covar_factor(v)

    def _eval_covar_matrix(self):
        cf = self.covar_factor
        if len(cf.size()) == 1:
            cf = cf.unsqueeze(-1)
        return cf @ cf.transpose(-1, -2) + self.const

    def add_prior(self):
        self.register_prior(
            "covar_factor_prior",
            self.covar_factor_prior,
            self._covar_factor_param,
            self._covar_factor_closure,
        )
