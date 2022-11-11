import numpy as np
import collections
from scipy.sparse import issparse, csr_matrix

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..dynamics import recover_dynamics_scvelo

# constant scaling spliced and unspliced counts across genes
# constant differential scaling spliced and unspliced counts aceoss genes
# differential scaling spliced and unspliced counts across genes (cell level peturbations post normalisation)
# poisson noise matrix added to data matrix (use feature mean and calibrate variance) 

# Adapted from https://github.com/theislab/scvelo/blob/master/scvelo/datasets.py#L308-L458
def add_noise(gene, adata, noise_level):

    unspliced = adata[:, gene].layers['unspliced'].toarray().flatten()
    spliced = adata[:, gene].layers['spliced'].toarray().flatten()

    unspliced += np.random.normal(
        scale=noise_level * np.percentile(unspliced, 99) / 10, size=len(unspliced))

    spliced += np.random.normal(
        scale=noise_level * np.percentile(spliced, 99) / 10, size=len(spliced))

    unspliced, spliced = np.clip(unspliced, 0, None), np.clip(spliced, 0, None)

    adata[:, gene].layers['unspliced'], adata[:, gene].layers['spliced'] = unspliced.reshape(-1,1), spliced.reshape(-1,1)

# Generate different kind of batch effects
def simulate_batch_effect(data, scaling=1, noise_level=1, scaling_range=[0.8, 1.2],
                          mode='add_noise', random_effect=True, n_jobs=-2, copy=True):

    adata = data.copy() if copy else data

    if not isinstance(scaling, (collections.Sequence, np.ndarray)):
        scaling = [scaling]
    
    for layer in ['spliced', 'unspliced']:
        adata.layers[layer] = adata.layers[layer].astype(float)
        if issparse(adata.layers[layer]):
            adata.layers[layer] = adata.layers[layer].toarray()

    if random_effect:
        scaling = np.random.uniform(scaling_range[0], scaling_range[1], adata.shape[1])
    
    if mode=='constant':
        scaling = scaling[0]
        adata.layers['spliced'] *= scaling
        adata.layers['unspliced'] *= scaling

    elif mode=='biconstant':
        scaling_s = scaling[0]
        scaling_u = scaling[1]
        adata.layers['spliced'] *= scaling_s
        adata.layers['unspliced'] *= scaling_u

    elif mode=='random':
        adata.layers['spliced'] *= scaling
        adata.layers['unspliced'] *= scaling
        
    elif mode=='add_noise':
        Parallel(n_jobs=n_jobs)(delayed(add_noise)(gene, adata, noise_level) for gene in tqdm(adata.var.index.values, 
                                                                                              desc="Adding noise to gene's expression", 
                                                                                              unit="genes"))

    for layer in ['spliced', 'unspliced']:
        adata.layers[layer] = csr_matrix(adata.layers[layer].astype(int))
    
    adata.uns['batch_simulation_params'] = {'mode': mode,
                                            'scaling': scaling,
                                            'scaling_range': scaling_range,
                                            'random_effect': random_effect,
                                            'noise_level': noise_level
                                                                } 
    
    if copy: return adata

# Generate multiple batches based off scaling factors
def simulate_batches(data, scalings=None, n_jobs=-2, copy=False):

    adata = data.copy() if copy else data

    # If scaling is not supplied
    if not scalings:
        raise ValueError('Provide tuple of spliced and unspliced scaling factors')

    if 'batch_effect_simulation' not in adata.uns.keys():
        adata.uns['batch_simulations'] = {}
    adata.uns['batch_simulations']['scalings'] = np.array(scalings)

    if copy: return adata

# Generate batches
def generate_batch(adata, return_dynamics=True):
    
    try: isinstance(adata.uns['batch_simulations']['scalings'], np.ndarray)
    except: raise ValueError('Run simulate_batches first.')

    num_batches = adata.uns['batch_simulations']['scalings'].shape[0]
    for i in range(num_batches):
        current_state = i
        # Biconstant mode allows gene wise scalings to be propagated as well (?)
        adata_ = simulate_batch_effect(adata, scaling=adata.uns['batch_simulations']['scalings'][i], 
                                       mode='biconstant', random_effect=False, copy=True)
        try:
            if return_dynamics:
                adata_.var = adata.uns['batch_simulations']['parameter_sets'][i]
                adata_.layers['fit_t'] = adata.uns['batch_simulations']['gene_latent_times'][i]
        except: pass
        yield adata_

# Return a singular batch 
def return_batch(adata, i, return_dynamics=True):

    try: isinstance(adata.uns['batch_simulations']['scalings'], np.ndarray)
    except: raise ValueError('Run simulate_batches first.')
    # Biconstant mode allows gene wise scalings to be propagated as well (?)
    adata_ = simulate_batch_effect(adata, scaling=adata.uns['batch_simulations']['scalings'][i], 
                                       mode='biconstant', random_effect=False, copy=True)
    try:
        if return_dynamics:
            adata_.var = adata.uns['batch_simulations']['parameter_sets'][i]
            adata_.layers['fit_t'] = adata.uns['batch_simulations']['gene_latent_times'][i]
    except: pass
    return adata_

# Recover dynamics for batches
def infer_batch_dynamics(adata, method='scvelo', **kwargs):

    if method=='scvelo':
        parameter_sets = []
        gene_latent_times = []
        for adata_ in generate_batch(adata, return_dynamics=False):
            var, fit_t = recover_dynamics_scvelo(adata_, **kwargs)
            parameter_sets.append(var)
            gene_latent_times.append(fit_t)

    adata.uns['batch_simulations']['parameter_sets'] = parameter_sets
    adata.uns['batch_simulations']['gene_latent_times'] = gene_latent_times