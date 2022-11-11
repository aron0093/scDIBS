import logging
logging.basicConfig(level = logging.INFO)
import collections
import numpy as np
import pandas as pd

from dtw import *
from tqdm.auto import tqdm

def prep_gene_latent_time(gene, params, key='fit'):

    gene_time = params[1][:, params[0].index.get_loc(gene)].flatten()
    if np.isnan(gene_time).all():
        return gene_time, np.nan, np.nan, np.nan

    # Record non zero cell indices for gene
    non_zero_indices = np.where([gene_time!=0])[1]
    gene_time = gene_time[non_zero_indices]
    
    # Insert switch time into array
    switch_time = params[0].loc[gene, '{}_t_'.format(key)]
    gene_time = np.append(gene_time, switch_time)
    
    # Map sorted indices to original non zero indices accounting for switch insertion
    sorted_indices = np.argsort(gene_time, kind='stable')
    gene_time = gene_time[sorted_indices]
    
    non_zero_indices = non_zero_indices[np.delete(sorted_indices, 
                                        np.where(sorted_indices==gene_time.shape[0]-1))]
        
    switch_index = np.where(gene_time==switch_time)[0][0]
    scaling = np.max(gene_time)
    gene_time /= scaling

    return gene_time, switch_index, non_zero_indices, scaling

def map_latent_time_density(data, parameter_set, genes=None, key='fit', 
                            open_begin=False, open_end=False, step_pattern='symmetric2', copy=False):

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
    if '{}_t'.format(key) not in adata.layers.keys():
        raise ValueError('Gene wise latent time is mising. Compute dynamics first!')
    
    adata.uns['density_dtw_alignment'] = {}
    adata.var['density_fit_t_'] = None
    adata.layers['density_fit_t'] = np.ones(adata.shape)*np.nan
    for gene in tqdm(genes, desc='Mapping genewise latent times', unit=' genes'):
        
        base_latent_time, base_switch_index, base_indices, base_scaling = prep_gene_latent_time(gene, parameter_set, key=key)
        mapped_latent_time, mapped_switch_index, mapping_indices, mapping_scaling = prep_gene_latent_time(gene, 
                                                                                         (adata.var, 
                                                                                          adata.layers['{}_t'.format(key)]),
                                                                                         key=key)
        
        if np.isnan(mapped_latent_time).all() or np.isnan(base_latent_time).all():
            adata.var.loc[gene, 'density_fit_t_'] = mapped_switch_index
            adata.layers['density_fit_t'][:, adata.var.index.get_loc(gene)] = adata.layers['{}_t'.format(key)][:, adata.var.index.get_loc(gene)]

        else:     
            alignment = dtw(mapped_latent_time,
                            base_latent_time,  
                            keep_internals=True, 
                            step_pattern=step_pattern, 
                            open_begin=open_begin, 
                            open_end=open_end
                           )

            warping_path = pd.DataFrame([alignment.reference[alignment.index2], alignment.index2, alignment.index1], 
                                         index=['mapped_latent_time', 'mapped_base_index', 'mapped_index']).T.groupby('mapped_index').mean().sort_index().reset_index()

            adata.var.loc[gene, 'density_fit_t_'] = warping_path.loc[warping_path.mapped_index==mapped_switch_index, 'mapped_latent_time'].values[0]
            mapped_latent_time = warping_path.loc[warping_path.mapped_index!=mapped_switch_index, 'mapped_latent_time'].values
            
            adata.layers['density_fit_t'][mapping_indices, adata.var.index.get_loc(gene)] = mapped_latent_time*base_scaling
            
            adata.layers['density_fit_t'][:, adata.var.index.get_loc(gene)] = \
            np.nan_to_num(adata.layers['density_fit_t'][:, adata.var.index.get_loc(gene)])
            
            #Save alignment object in .uns
            adata.uns['density_dtw_alignment'][gene] = alignment

    if copy: return adata