#!/usr/bin/env python3

"""
MultiTaskGPICM is a subclass of MultiTaskGP that implements a multi-task Gaussian Process model with an Intrinsic Coregionalization Model (ICM) structure.
Args:
    train_X (Tensor): Training input data.
    train_Y (Tensor): Training output data.
    task_feature (int): Index of the task feature.
    train_Yvar (Tensor, optional): Observation noise for training data. Defaults to None.
    mean_module (Module, optional): Mean module for the GP. Defaults to None.
    covar_module (Module, optional): Covariance module for the GP. Defaults to None.
    task_covar_module (Module, optional): Task covariance module for the GP. Defaults to None.
    likelihood (Likelihood, optional): Likelihood for the GP. Defaults to None.
    task_covar_prior (Prior, optional): Prior for the task covariance. Defaults to None.
    output_tasks (List[int], optional): List of output tasks. Defaults to None.
    rank (int, optional): Rank of the task covariance matrix. Defaults to None.
    input_transform (InputTransform, optional): Input transform for the GP. Defaults to None.
    outcome_transform (OutcomeTransform, optional): Outcome transform for the GP. Defaults to None.
Methods:
    forward(x: Tensor) -> MultivariateNormal:
        Computes the forward pass of the model, returning a MultivariateNormal distribution.
        Args:
            x (Tensor): Input tensor.
        Returns:
            MultivariateNormal: The predicted multivariate normal distribution.
"""


from typing import List
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.module import Module
from gpytorch.priors.prior import Prior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
import torch


class MultiTaskGPICM(MultiTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        task_covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | None = None,
        output_tasks: List[int] | None = None,
        rank: int | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            task_feature,
            train_Yvar,
            mean_module,
            covar_module,
            likelihood,
            task_covar_prior,
            output_tasks,
            rank,
            input_transform,
            outcome_transform,
        )
        if task_covar_module is not None:
            self.task_covar_module = task_covar_module

    def forward(self,x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x_basic, task_idcs = self._split_inputs(x)
        # Compute base mean and covariance
        mean_x = self.mean_module(x_basic)
        mean_x = mean_x[...,torch.arange(task_idcs.size(-2)),task_idcs.reshape(-1,task_idcs.size(-2))[0]]
        covar_x = self.covar_module(x_basic)
        # Compute task covariances
        covar_i = self.task_covar_module(task_idcs)
        # Combine the two in an ICM fashion
        covar = covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)
        
