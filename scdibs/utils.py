import os, contextlib
import numpy as np

# Calculates the inverse of x where x!=0.
def inv(x):
    x_inv = 1 / x * (x != 0)
    return x_inv

# Supress unwanted print messages
# WARNING: All standard output will be supressed
def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper

# Extract params and genewise latent time
def extract_params(adata, key='fit'):
    var = adata.var.copy()
    var = var.loc[:, [col for col in var.columns if col.startswith(key)]]
    return (var, adata.layers['{}_t'.format(key)].copy())

# Rescale gene latent time by subtracting switch time and dividing by absolute max
def rescale_latent_time(params, key='fit'):

    var, gene_time = params

    gene_time = gene_time - var.loc[:, '{}_t_'.format(key)].values.reshape(1,-1)
    gene_time = gene_time/np.abs(gene_time).max(axis=0)

    return gene_time

# Estimate scaling factor for switch times
def estimate_scaling(fit_t_base, fit_t_mapped):
    return fit_t_base/fit_t_mapped