import logging
logging.basicConfig(level = logging.INFO)

import numpy as np
from ..dynamics import recover_dynamics_scvelo, compute_dynamics_scvelo

# Produce spliced and unspliced counts using external parameters (but pre-computed latent time)
def apply_cross_dynamics_scvelo(data, parameter_set, genes=None, copy=False, **kwargs):

    adata = data.copy() if copy else data

    # Parameter set made using recover_dynamics_scvelo
    var = parameter_set[0].copy()

    # Check if latent time exist for common genes
    genes_= np.intersect1d(var.index.values, adata.var.index.values)
    if genes is None:
        genes = genes_
        if len(genes)==0:
            raise ValueError('No overlapping genes found!')
    elif isinstance(genes, (collections.Sequence, np.ndarray)):
        genes = np.intersect1d(genes, genes_)
    elif isinstance(genes, str):
        genes = [genes]
    else:
        raise ValueError('Improper specification of genes.')

    # Native fit of gene latent times
    if 'fit_t' not in adata.layers.keys():
        _, fit_t = recover_dynamics_scvelo(adata, kwargs)
    
    adata.layers['unspliced_cross'], adata.layers['spliced_cross'] = np.ones(adata.shape)*np.nan, np.ones(adata.shape)*np.nan

    #TODO: Parallelize
    logging.info('Computing cross dynamics')
    for j, gene in enumerate(adata.var_names):
        if gene in genes:
            adata.layers['unspliced_cross'][:, j], adata.layers['spliced_cross'][:, j] = compute_dynamics_scvelo(var, adata.layers['fit_t'], gene, key='fit')

    # Save cross fit params in .var
    var.columns = ['cross_{}'.format(col) for col in var.columns]
    adata.var = adata.var.merge(var, right_index=True, left_index=True, how='left')

    if copy: return adata
