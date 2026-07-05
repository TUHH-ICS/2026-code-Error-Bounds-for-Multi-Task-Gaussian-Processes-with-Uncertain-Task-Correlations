#!/usr/bin/env python3

"""
This module provides utility functions for working with control systems, 
specifically for laser models and controllers. It includes functions for 
calculating H2 norms, building laser model chains, and creating PI and P controllers.

Functions:
    get_nh2(param, G, bounds, K_typ="PI"):
        Computes the negative H2 norm for a given set of parameters and system.

    h2_norm(ss: StateSpace):
        Calculates the H2 norm of a given state-space system.

    build_laser_model(num_laser: int = 1, disturbance: float | None = None):
        Constructs a laser model with the specified number of lasers and disturbance.

    pi_controller(params):
        Creates a list of PI controllers based on the given parameters.

    p_controller(params):
        Creates a list of P controllers based on the given parameters.

    get_closed_loop(params, G, C="PI"):
        Constructs a closed-loop system with the given parameters and controller type.
"""

from control import gram, summing_junction, interconnect, tf2ss, tf
from control import StateSpace
from numpy import trace, sqrt, minimum
from . models import get_disturbance_filter, get_laser_model, get_reference_filter
from botorch.utils.transforms import unnormalize
from slycot.exceptions import SlycotError
import torch
from joblib import Parallel, delayed


def safe_eval(i, param_i, G, K_typ):
    flag = False
    while not flag:
        try:
            val = h2_norm(get_closed_loop(param_i, G, K_typ))
            return i, torch.tensor(val)
        except SlycotError:
            print(f"SLICOT ERROR at index {i}, retrying...")
            param_i += 1e-8 * torch.ones_like(param_i)

def get_nh2(param, G, bounds, K_typ, n_jobs=-1):
    # Ensure n_jobs is an integer
    param = unnormalize(param, bounds).round(decimals=7)
    if not isinstance(n_jobs, int):
        try:
            n_jobs = int(n_jobs)
        except Exception:
            n_jobs = -1

    num_evals = param.shape[0]
    param_np = param.detach().numpy()

    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(safe_eval)(i, param_np[i, ...], G, K_typ)
        for i in range(num_evals)
    )

    # Sort results by index
    results.sort()
    vals = torch.stack([val for _, val in results])
    return -vals.view(-1, 1)  

def get_nh22(param, G, bounds, K_typ="PI"):
    param = unnormalize(param, bounds).round(decimals=7)
    num_evals = param.size(0)
    vals = torch.zeros(num_evals, 1)
    for i in range(num_evals):
        flag = False
        while not flag:
            try:
                vals[i] = torch.tensor(
                    h2_norm(get_closed_loop(param[i, ...].detach().numpy(), G, K_typ))
                )
                flag = True
            except SlycotError:
                flag = False
                print("\nSLYCOT ERROR\n")
                param[i, ...] += 1e-8 * torch.ones_like(param[i, ...])
    return -vals


MAX_H2 = 50.


def h2_norm(ss: StateSpace):
    B = ss.B
    try:
        Wo = gram(ss, "o")
    except ValueError as e:
        print(f"Got error {e}.\n Returned max H2 value:{MAX_H2}")
        return MAX_H2
    H2 = minimum(sqrt(trace(B.T @ Wo @ B)), MAX_H2)
    return H2


def build_laser_model(num_laser: int = 1, disturbance: float | None = None):
    Fr = get_reference_filter(disturbance)
    sumblk = []
    G_list = []
    Fd_list = []
    for i in range(num_laser):
        Fd = get_disturbance_filter(disturbance)
        G = get_laser_model()
        Fd.input_labels = f"w({i})"
        Fd.output_labels = f"d({i})"
        Fd.name = f"Fd({i})"
        G.input_labels = f"u({i})"
        G.output_labels = f"phi({i})"
        G.name = f"G({i})"
        if i == 0:
            sumblk.extend(
                [
                    summing_junction(inputs=["phi(0)", "d(0)"], output=f"y(0)"),
                    summing_junction(inputs=["r", "-y(0)"], output=f"e(0)"),
                ]
            )
            Fd_list.append(Fd)
            G_list.append(G)
        else:
            Fd_list.append(Fd)
            G_list.append(G)
            sumblk.extend(
                [
                    summing_junction(inputs=[f"phi({i})", f"d({i})"], output=f"y({i})"),
                    summing_junction(
                        inputs=[f"y({i-1})", f"-y({i})"], output=f"e({i})"
                    ),
                ]
            )

    inputs = (
        [f"u({i})" for i in range(num_laser)]
        + ["r"]
        + [f"w({i})" for i in range(num_laser)]
    )
    outputs = [f"e({i})" for i in range(num_laser)] + [f"y({num_laser-1})"]
    Glaser = interconnect(sumblk + G_list + Fd_list, inputs=inputs, outputs=outputs)

    inputs[num_laser] = "wr"
    outputs[-1] = "z"
    sumblk = summing_junction(inputs=[f"-y({num_laser-1})", "r"], outputs="z")
    GlaserChain = interconnect([Glaser, sumblk, Fr], inputs=inputs, outputs=outputs)
    return GlaserChain


k_phi = 330000


def pi_controller(params):
    s = tf("s")
    num_c = int(params.shape[-1] / 2)
    params = params.reshape(num_c, 2) / k_phi
    C = []
    c = 0
    for i in params:
        Ct = tf2ss(i[0] + i[1] / s)
        Ct.name = f"C({c})"
        Ct.input_labels = f"e({c})"
        Ct.output_labels = f"u({c})"
        C.append(Ct)
        c += 1
    return C


def p_controller(params):
    num_c = params.shape[-1]
    params = params.reshape(num_c, 1) / k_phi
    C = []
    c = 0
    for i in params:
        Ct = tf(i[0], [1.0])
        Ct.name = f"C({c})"
        Ct.input_labels = f"e({c})"
        Ct.output_labels = f"u({c})"
        C.append(Ct)
        c += 1
    return C


def get_closed_loop(params, G, C="PI"):
    if C == "PI":
        K = pi_controller(params)
    elif C == "P":
        K = p_controller(params)
    else:
        raise ValueError(f"Unknown controller type '{C}'. Expected 'PI' or 'P'.")
    num_c = len(K)
    inputs = [f"w({i})" for i in range(num_c)] + ["wr"]
    return interconnect(K + [G], inputs=inputs, outputs=["z"])
