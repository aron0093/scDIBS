import logging
logging.basicConfig(level = logging.INFO)
import collections
import numpy as np

from ..utils import estimate_scaling

from tqdm.auto import tqdm

# Match steady time to rescale gene wise latent time
def match_steady_state(data, parameter_set, genes=None, copy=False, **kwargs):

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
        raise ValueError('Gene wise latent time is mising. Compute dynamics first!')

    adata.var['steady_state_rescaling'] = None
    adata.var['ssr_fit_t_'] = None
    adata.layers['ssr_fit_t'] = np.ones(adata.shape)*np.nan
    for gene in tqdm(genes, desc='Mapping genewise latent times', unit=' genes'):
        
        adata.var.loc[gene, 'steady_state_rescaling'] = estimate_scaling(var.loc[gene, 'fit_t_'],
                                                                         adata.var.loc[gene, 'fit_t_'])
        adata.var.loc[gene, 'ssr_fit_t_'] = adata.var.loc[gene, 'fit_t_']*adata.var.loc[gene, 'steady_state_rescaling']

        adata.layers['ssr_fit_t'][:, adata.var.index.get_loc(gene)] = adata.layers['fit_t'][:, adata.var.index.get_loc(gene)]*adata.var.loc[gene, 'steady_state_rescaling']
    
    # Save cross fit params in .var
    var.columns = ['base_{}'.format(col) for col in var.columns]
    adata.var = adata.var.merge(var, right_index=True, left_index=True, how='left')
    
    if copy: return adata

# Steady state matching for subsamples or batches
def match_steady_state_sets(data, key='batch_simulations', genes=None, copy=False, **kwargs):

    adata = data.copy() if copy else data

    # Load parameter sets
    vars_ = adata.uns[key]['parameter_sets']
    fit_ts = adata.uns[key]['gene_latent_times']

    ssr_fit_ts = []
    for i in range(len(vars_)):

        # Check if latent time exist for common genes
        genes_= np.intersect1d(adata.var.index.values, vars_[i].index.values)
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
        vars_[i]['steady_state_rescaling'] = None
        ssr_fit_t_ = np.ones(fit_ts[i].shape)*np.nan
        for gene in genes:
            # adata here is the base unlike previous function
            vars_[i].loc[gene, 'steady_state_rescaling'] = estimate_scaling(adata.var.loc[gene, 'fit_t_'],
                                                                            vars_[i].loc[gene, 'fit_t_'])

            ssr_fit_t_[:, vars_[i].index.get_loc(gene)] = fit_ts[i][:, vars_[i].index.get_loc(gene)]*vars_[i].loc[gene, 'steady_state_rescaling']
        ssr_fit_ts.append(ssr_fit_t_)
    adata.uns[key]['ssr_gene_latent_times'] = ssr_fit_ts
    
    if copy: return adata

        




