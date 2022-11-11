import numpy as np
from numpy.random import default_rng
from ..dynamics import recover_dynamics_scvelo

# Make random subsets of scRNAseq data with or without replacement
# Make the list of cell indices
def subsampling(data, num_subsamples=5, subsample_size_ratio=0.8, bootstrap=False, random_state=0, copy=False):

    adata = data.copy() if copy else data

    # Set seed fo reproducibility
    rng = default_rng(random_state)

    # sub-sample size; subsample_size_ratio only relevant for sampling without replacement
    adata_shape = adata.shape[0]
    if not bootstrap:
        subsample_size = int(adata_shape*subsample_size_ratio)
        subsample_index_array = np.empty((num_subsamples, subsample_size), dtype=int)
        for i in range(num_subsamples):
            subsample_index_array[i] = rng.choice(adata_shape, size=subsample_size, replace=False)
    else:
        subsample_size = adata_shape
        subsample_index_array = rng.choice(adata_shape, size=(num_subsamples, subsample_size), replace=True)

    if 'subsampling' not in adata.uns.keys():
        adata.uns['subsampling'] = {}

    adata.uns['subsampling']['params'] = {'subsample_size': subsample_size,
                                       'subsample_size_ratio': subsample_size_ratio,
                                       'bootstrap': bootstrap,
                                       'random_state': random_state
                                      }
    adata.uns['subsampling']['index_array'] = subsample_index_array

    if copy: return adata

# Generate subsamples
def generate_subsample(adata, return_dynamics=True):

    try: isinstance(adata.uns['subsampling']['index_array'], np.ndarray)
    except: raise ValueError('Run subsampling first.')
    
    num_subsamples = adata.uns['subsampling']['index_array'].shape[0]
    for i in range(num_subsamples):
        current_state = i
        adata_ = adata[adata.uns['subsampling']['index_array'][i]].copy()
        try:
            if return_dynamics:
                adata_.var = adata.uns['subsampling']['parameter_sets'][i]
                adata_.layers['fit_t'] = adata.uns['subsampling']['gene_latent_times'][i]
        except: pass
        yield adata_

# Return a singular subsample 
def return_subsample(adata, i, return_dynamics=True):

    try: isinstance(adata.uns['subsampling']['index_array'], np.ndarray)
    except: raise ValueError('Run subsampling first.')
    
    adata_ = adata[adata.uns['subsampling']['index_array'][i]].copy()
    try:
        if return_dynamics:
            adata_.var = adata.uns['subsampling']['parameter_sets'][i]
            adata_.layers['fit_t'] = adata.uns['subsampling']['gene_latent_times'][i]
    except: pass

    return adata_

# Recover dynamics for subsamples
def infer_subsample_dynamics(adata, method='scvelo', **kwargs):

    if method=='scvelo':
        parameter_sets = []
        gene_latent_times = []
        for adata_ in generate_subsample(adata, return_dynamics=False):
            var, fit_t = recover_dynamics_scvelo(adata_, **kwargs)
            parameter_sets.append(var)
            gene_latent_times.append(fit_t)

    adata.uns['subsampling']['parameter_sets'] = parameter_sets
    adata.uns['subsampling']['gene_latent_times'] = gene_latent_times