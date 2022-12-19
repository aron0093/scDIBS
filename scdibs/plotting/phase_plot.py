import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scipy.sparse import issparse

from ..dynamics import compute_dynamics_scvelo
from ..utils import extract_params

def plot_phase_scvelo(params, gene, key='fit', color=None, ax=None):

    u, s = compute_dynamics_scvelo(params[0], params[1], gene, key=key)
    ax = sns.scatterplot(u, s, color=color, ax=ax)
    
    u = u[np.argsort(params[1][:, params[0].index.get_loc(gene)], kind='stable')]
    s = s[np.argsort(params[1][:, params[0].index.get_loc(gene)], kind='stable')]
    
    ax.plot(u,s)

    return ax

def plot_comparitive_phase_scvelo(params_base, params_mapped, gene, key='fit', rescale=False, ax=None):

    vars_ = {'base': params_base[0], 
             'mapped': params_mapped[0]}
    gene_times = {'base': params_base[1], 
                  'mapped': params_mapped[1]}

    u, s = {}, {}
    for i in ('base', 'mapped'):
        for j in ('base', 'mapped'):
            u_, s_ = compute_dynamics_scvelo(vars_[i].loc[[gene]], 
                                             gene_times[j][:, [vars_[j].index.get_loc(gene)]], 
                                             gene, key=key)

            if rescale:
                u_ = u_/max(u_)
                s_ = s_/max(s_)

            u_ = u_[np.argsort(gene_times[j][:, vars_[j].index.get_loc(gene)], kind='stable')]
            s_ = s_[np.argsort(gene_times[j][:, vars_[j].index.get_loc(gene)], kind='stable')]
    
            u['_'.join((i,j))], s['_'.join((i,j))] = u_, s_

    ax = sns.scatterplot(u['base_base'], s['base_base'], ax=ax)
    ax.plot(u['base_base'],s['base_base'])

    ax = sns.scatterplot(u['mapped_mapped'], s['mapped_mapped'], ax=ax)
    ax.plot(u['mapped_mapped'],s['mapped_mapped'])

    for n in range(gene_times['mapped'].shape[0]):
         ax.plot([u['mapped_mapped'][n], u['base_mapped'][n]], 
                 [s['mapped_mapped'][n], s['base_mapped'][n]], 
                 color='grey')
    

    return ax

def plot_latent_expression(adata, gene, key='fit', color=None, ax=None):

    if issparse(adata.X):
        ax = sns.scatterplot(adata.layers['{}_t'.format(key)][:, adata.var.index.get_loc(gene)], 
                    adata.X.toarray()[:, adata.var.index.get_loc(gene)], 
                    color=color, ax=ax
                    )
    else:
        ax = sns.scatterplot(adata.layers['{}_t'.format(key)][:, adata.var.index.get_loc(gene)], 
                    adata.X[:, adata.var.index.get_loc(gene)], 
                    color=color
                    ) 
    ax.vlines(adata.var.loc[gene, '{}_t_'.format(key)], ymin=0, ymax=ax.get_ylim()[1], color=color)

    return ax

def plot_latent_distribution(adata, gene, key='fit', color=None, ax=None):

    ax = sns.histplot(adata.layers['{}_t'.format(key)][:, adata.var.index.get_loc(gene)],
                      color=color, ax=ax
                      )
    ax.vlines(adata.var.loc[gene, '{}_t_'.format(key)], ymin=0, ymax=ax.get_ylim()[1], color=color)

    return ax

def plot_phase_plot_panel(adata, genes=None, key='fit', color=None, figsize=(15, 3.5)):

    fig, axs = plt.subplots(ncols=3, nrows=len(genes), figsize=(15, len(genes)*3.5))

    for i, gene in enumerate(genes):
        j = i*3

        plot_phase_scvelo(extract_params(adata), gene, key=key, color=color, ax=axs.flat[j])
        plot_latent_expression(adata, gene, key=key, color=color, ax=axs.flat[j+1])
        plot_latent_distribution(adata, gene, key=key, color=color, ax=axs.flat[j+2])

    return fig, axs

