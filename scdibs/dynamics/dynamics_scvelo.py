import numpy as np
import pandas as pd
import collections

import scvelo as scv
from ..utils import inv

def vectorize(t, t_, alpha, beta, gamma=None, alpha_=0, u0=0, s0=0):
    o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = latent_unspliced_scvelo(alpha, beta, t_, u0)
    s0_ = latent_spliced_scvelo(alpha, beta, gamma if gamma is not None else beta / 2, t_, s0, u0)

    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)
    return tau, alpha, u0, s0

# Calculates u(t) with t the time for one or multiple observations.
def latent_unspliced_scvelo(alpha, beta, t, u0):
    expu = np.exp(-beta * t)
    return u0 * expu + alpha / beta * (1 - expu)

# Calculates s(t) with t the time for one or multiple observations.
def latent_spliced_scvelo(alpha, beta, gamma, t, u0, s0):
    expu, exps = np.exp(-beta * t), np.exp(-gamma * t)
    return s0 * exps + (alpha / gamma) * (1 - exps) + (alpha - beta * u0) / inv(gamma - beta) * (exps - expu)

# Scvelo dynamics
def recover_dynamics_scvelo(adata, smoothen=False, n_jobs=-1):
    if smoothen:
        scv.pp.moments(adata)
    scv.tl.recover_dynamics(adata, n_jobs=n_jobs)

    return adata.var, adata.layers['fit_t']

def compute_dynamics_scvelo(var, fit_t, gene, key='fit'):

    idx = var.index.get_loc(gene)

    alpha = var.loc[gene, key + "_alpha"]
    beta = var.loc[gene, key + "_beta"]

    scaling = 1
    if f"{key}_scaling" in var.columns:
        scaling = var.loc[gene, key + "_scaling"]
    beta *= scaling   
    
    gamma = var.loc[gene, key + "_gamma"]
    t_ = var.loc[gene, key + "_t_"]
    
    if key + "_u0" in var.columns:
        u0_offset, s0_offset = var.loc[gene, key + "_u0"], var.loc[gene, key + "_s0"]
    else:
        u0_offset, s0_offset = 0, 0

    t = fit_t[:, idx]

    tau, alpha_, u0, s0 = vectorize(t, t_, alpha, beta, gamma)
    ut, st = latent_unspliced_scvelo(alpha_, beta, tau, u0), latent_spliced_scvelo(alpha_, beta, gamma, tau, u0, s0)
    ut, st = ut * scaling + u0_offset, st + s0_offset
    
    return ut, st
                









