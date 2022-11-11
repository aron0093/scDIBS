import numpy as np
from sklearn.preprocessing import minmax_scale, scale
from scvelo.preprocessing import neighbors

from ..utils import rescale_latent_time

def compute_temporal_representation(data, likelihood_threshold=0.5, selected_genes= None, method='umap', 
                                    key='fit', rescale=True, n_jobs=-1, random_state=0, copy=False):

    adata = data.copy() if copy else data

    # Check if gene wise latent time exists
    try: assert adata.layers['{}_t'.format(key)].shape
    except: raise ValueError('Gene wise latent time not found')

    # Filter genes by likelihood
    if selected_genes is None:
        selected_genes = adata.var.index[adata.var.fit_likelihood >= likelihood_threshold].values
        
    gene_time = adata[:, selected_genes].layers['{}_t'.format(key)]
    
    # Rescale gene wise latent times by subtracting steady state and dividing by max
    if rescale:
        params = (adata.var.loc[selected_genes], gene_time)
        gene_time = rescale_latent_time(params, key=key)

    # Store rescaled gene latent time in embedding 
    adata.obsm['X_latent'] = gene_time
    adata.obsm['X_latent'] = adata.obsm['X_latent'].T[~np.isnan(adata.obsm['X_latent'].T).any(axis=1)].T

    adata.var['temporal_knn_selected_genes'] = False
    adata.var.loc[selected_genes, 'temporal_knn_selected_genes'] = True

    if copy: return adata

def compute_temporal_knn(data, n_neighbors=30, method='umap', n_jobs=-1, random_state=0, copy=False):

    adata = data.copy() if copy else data

    if 'X_latent' not in adata.obsm.keys():
        raise ValueError('Compute temporal representation first.!')
        
    # Compute neighborhood
    neighbors(adata, n_neighbors=n_neighbors, n_pcs=None, use_rep='latent', 
              use_highly_variable=True, knn=True, random_state=random_state, 
              method=method, metric='euclidean', 
              metric_kwds=None, num_threads=n_jobs, copy=False)

    if copy: return adata