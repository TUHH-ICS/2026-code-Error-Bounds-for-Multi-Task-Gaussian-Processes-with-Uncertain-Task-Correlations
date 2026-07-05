#!/usr/bin/env python3

"""
This module contains various classes for defining and working with different types of functions, including RKHS functions, 
multi-task RKHS functions, and several synthetic benchmark functions for optimization.
Classes:
    RKHSFunction:
        Represents a Reproducing Kernel Hilbert Space (RKHS) function.
        Methods:
            __init__(self, bounds, B, n, ns, kernel):
                Initializes the RKHS function with given parameters.
            f(self, x):
                Evaluates the RKHS function at given points.
            plot(self):
                Plots the RKHS function over a grid of points.
    MTRKHSFunction:
        Represents a multi-task RKHS function.
        Methods:
            __init__(self, bounds, B, n, ns, kernel, num_tsks):
                Initializes the multi-task RKHS function with given parameters.
            cov(self, x1, x2, t1, t2):
                Computes the covariance between two sets of points and tasks.
            f(self, x, t):
                Evaluates the multi-task RKHS function at given points and tasks.
            plot(self):
                Plots the multi-task RKHS function over a grid of points.
    LbSync:
        Represents a laser beam synchronization function.
        Methods:
            __init__(self, Ktyp, num_lasers, num_tsks, disturbance):
                Initializes the laser beam synchronization function with given parameters.
            plot(self):
                Plots the laser beam synchronization function over a grid of points.
            f(self, x, t):
                Evaluates the laser beam synchronization function at given points and tasks.
"""

import os

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import torch
from torch import Tensor
from gpytorch.kernels import RBFKernel, Kernel
import matplotlib.pyplot as plt
from utils.plant.utils import build_laser_model, get_nh2
from botorch.utils.transforms import normalize, unnormalize

class RKHSFunction:
    def __init__(self, bounds = torch.tensor([[0],[1]]), B=2, n = 20, ns = 1000, kernel:Kernel = RBFKernel()):
        self.bounds = bounds
        self.dim = bounds.size(-1)
        self.ns = ns
        self.kernel = kernel
        self.kernel._set_lengthscale(torch.tensor([[0.1]*self.dim]))
        self.xt = torch.linspace(bounds[0,0], bounds[1,0], n).view(-1, 1)
        self.alpha_tilde = torch.randn(n, 1)
        self.alpha = self.alpha_tilde / torch.sqrt(self.alpha_tilde.T @ self.kernel(self.xt, self.xt) @ self.alpha_tilde) * B

    def f(self, x):
        return self.kernel(x, self.xt) @ self.alpha
    
    def plot(self):
        x_grid = torch.linspace(self.bounds[0,0], self.bounds[0,1], self.ns).view(-1, 1)
        y_grid = self.f(x_grid)
        plt.plot(x_grid.detach().numpy(), y_grid.detach().numpy())
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Plot of f(x)')
        plt.show()

class MTRKHSFunction:
    def __init__(self, bounds = torch.tensor([[0],[1]]), B=2, n = 5, ns = 300, kernel:Kernel = RBFKernel(), num_tsks = 2):
        self.bounds = bounds
        self.ns = ns
        self.B = B
        self.num_tsks = num_tsks  
        self.base_kernel = kernel
        self.base_kernel._set_lengthscale(torch.tensor([[0.08]]))
        self.index_kernel = torch.tensor([[1., 0.9], [0.9, 1.]])
        self.xt = torch.rand(n*num_tsks,1)
        self.tt = torch.hstack([torch.zeros(n), torch.ones(n)]).to(dtype=torch.int64).view(-1, 1)
        self.alpha_tilde = torch.randn(n*num_tsks, 1)
        self.alpha = self.alpha_tilde / torch.sqrt(self.alpha_tilde.T @ self.cov(self.xt,self.xt,self.tt,self.tt) @ self.alpha_tilde) * B
        self.dim = self.bounds.size(-1)

    def cov(self, x1, x2, t1, t2):
        return self.base_kernel(x1, x2) * self.index_kernel[t1, t2.T]

    def f(self, x, t):
        if not isinstance(t,Tensor):
            t = torch.tensor([t])
        if t.size(0)!= x.size(0):
            t = t.repeat(x.size(0),1)
        return self.cov(unnormalize(x,self.bounds),self.xt,t.view(-1,1),self.tt) @ self.alpha
    
    def plot(self):
        if self.dim == 1:
            with torch.no_grad():
                x_grid = torch.linspace(self.bounds[0,0], self.bounds[1,0], self.ns).view(-1, 1)
                t_grid = torch.hstack([torch.zeros(self.ns,1), torch.ones(self.ns,1)]).to(dtype=torch.int64)
                y_grid = torch.hstack([self.f(x_grid,t_grid[:,i]) for i in range(self.num_tsks)])
            fig, ax = plt.subplots()
            for i in range(self.num_tsks):
                ax.plot(x_grid.detach().numpy(), y_grid[:, i].detach().numpy(), label=f'f_{i}')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Plot of f(x)')
            ax.legend()
            plt.show()
        else:
            pass


class LbSync:
    def __init__(self, Ktyp = "PI", num_lasers = 2, num_tsks = 2, disturbance = .1) -> None:
        Kp_max = 3e1
        Kp_min = 2e-1
        Ki_max = 3e1
        Ki_min = 0
        self.num_tsks = num_tsks
        self.max_disturbance = 0.

        if Ktyp == "PI":
            self.bounds=torch.tensor([[Kp_min,Ki_min],[Kp_max,Ki_max]]).repeat(1,num_lasers)
        else:
            self.bounds=torch.tensor([[Kp_min],[Kp_max]]).repeat(1,num_lasers)

        self.dim = self.bounds.size(-1)
        G = [build_laser_model(num_lasers, disturbance = 0. if i == 0 else disturbance) for i in range(num_tsks)]
        self.obj = [lambda param, G=G[i]: get_nh2(param,G,self.bounds,Ktyp) for i in range(num_tsks)]
    
    def plot(self):
        if self.dim == 1:
            x = torch.linspace(0.,1., 100).view(-1,1)
            fig, ax = plt.subplots()
            for i in range(len(self.obj)):
                y = self.f(x, i) 
                ax.plot(x.squeeze(), y.squeeze(),label=f'f_{i}')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Plot of f(x)')
            ax.legend()
            plt.show()
        else:
            pass

    def f(self, x, t):
        return self.obj[t](x)
    
    

