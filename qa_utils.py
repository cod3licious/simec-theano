import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import spearmanr
from scipy.spatial.distance import pdist
from matplotlib import rcParams
from utils import get_colors
clist = [color['color'] for color in list(rcParams['axes.prop_cycle'])]


def shepards_plot(S_org, D_emb, col=None):
    """
    Shepard's plot to check the quality of an embedding with respect to the preserved distances/similarities

    Inputs:
        - S_org: original similarity matrix (NxN)
        - D_emb: distances in the embedding space (a corresponding NxN matrix)
        - col (optional): vector with numbers that can be turned into color specifications (like used for plotting the dataset)
    Returns:
        - rho: the spearman rank corrcoef of the Shepard's plot as a quality measure of the embedding
    """
    assert S_org.shape == D_emb.shape, 'Input matrices need to be of the same shape'
    assert S_org.shape[0] == S_org.shape[1], 'matrices need to be square'
    # to make the results comparable across methods, normalize S and D to be between 0 and 1
    S = S_org - np.min(S_org)
    D = D_emb - np.min(D_emb)
    S /= np.max(S)
    D /= np.max(D)
    # if matrices are too large, subsample
    if S.shape[0] > 500:
        idx = np.random.permutation(S.shape[0])[:500]
        S = S[np.ix_(idx, idx)]
        D = D[np.ix_(idx, idx)]
        if col is not None:
            col = col[idx]
    # plot the original Shepards diagram
    iu = np.triu_indices(S.shape[0], 1)
    sims = S[iu]
    dists = D[iu]
    rho = spearmanr(sims, dists)[0]
    if col is not None:
        c = np.tile(col, (S.shape[0], 1))
    else:
        c = clist[0]
    plt.figure()
    plt.scatter(D, S, c=c, s=2, linewidth=0.)
    plt.grid()
    plt.xlabel('distances in the embedding space', fontsize=14)
    plt.ylabel('original similarities', fontsize=14)
    plt.title("Shepard's Plot ($\\rho = %.3f$)" % rho, fontsize=16)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # plt.tight_layout()
    return rho


def original_NN_error(S_org, D_emb):
    """
    Inputs:
        - S_org: original similarity matrix (NxN)
        - D_emb: distances in the embedding space (a corresponding NxN matrix)
    Returns:
        - err_mean, err_std, s: mean and std of original NN error as well as corresponding similarities
    """
    assert S_org.shape == D_emb.shape, 'Input matrices need to be of the same shape'
    assert S_org.shape[0] == S_org.shape[1], 'matrices need to be square'
    # to make the results comparable across methods, normalize S and D to be between 0 and 1
    S = S_org - np.min(S_org)
    D = D_emb - np.min(D_emb)
    S /= np.max(S)
    D /= np.max(D)
    # get index for k nearest neighbors in original and embedding space
    NN_O = np.ogrid[:S.shape[0], :S.shape[1]]
    NN_O[1] = np.fliplr(np.argsort(S))
    NN_E = np.ogrid[:D.shape[0], :D.shape[1]]
    NN_E[1] = np.argsort(D)
    # compute raw error for every NN position
    E = D[NN_O] - D[NN_E]
    # points too close don't count, thats a problem of the embedding error
    E[E < 0] = 0.
    # accumulate the sum for all neighborhood sizes
    w = np.ones(S.shape[1]) / np.arange(1, S.shape[1] + 1)
    E_acc = w * np.add.accumulate(E, 1)
    # take mean and std across data points --> 2 arrays for all values of k
    err_mean, err_std = np.mean(E_acc, 0), np.std(E_acc, 0)
    return err_mean, err_std


def embedding_NN_error(S_org, D_emb):
    """
    Inputs:
        - S_org: original similarity matrix (NxN)
        - D_emb: distances in the embedding space (a corresponding NxN matrix)
    Returns:
        - err_mean, err_std, d: mean and std of embedding NN error as well as corresponding distances
    """
    assert S_org.shape == D_emb.shape, 'Input matrices need to be of the same shape'
    assert S_org.shape[0] == S_org.shape[1], 'matrices need to be square'
    # to make the results comparable across methods, normalize S and D to be between 0 and 1
    S = S_org - np.min(S_org)
    D = D_emb - np.min(D_emb)
    S /= np.max(S)
    D /= np.max(D)
    # get index for k nearest neighbors in original and embedding space
    NN_O = np.ogrid[:S.shape[0], :S.shape[1]]
    NN_O[1] = np.fliplr(np.argsort(S))
    NN_E = np.ogrid[:D.shape[0], :D.shape[1]]
    NN_E[1] = np.argsort(D)
    # compute raw error for every NN position
    E = S[NN_O] - S[NN_E]
    # "too" similar is ok - the original error deals with this
    # (but don't just set them to 0 or it'll ruin the mean)
    E[E < 0] = 0.
    # accumulate the sum for all neighborhood sizes
    w = np.ones(S.shape[1]) / np.arange(1, S.shape[1] + 1)
    E_acc = w * np.add.accumulate(E, 1)
    # take mean and std across data points --> 2 arrays for all values of k
    err_mean, err_std = np.mean(E_acc, 0), np.std(E_acc, 0)
    return err_mean, err_std


def plot_errors(err_dict, title, xlabel='k', ylabel='error', xlim=None, ylim=None, models=[]):
    """
    Inputs:
        - err_dict: a dict with {'model':{'err_mean':[], 'err_min':[], 'err_max':[]}} with mean, min and max error values for multiple models
    """
    plt.figure()
    if not models:
        models = sorted(err_dict.keys())
    colors = get_colors(len(models))
    x = range(1, len(err_dict[err_dict.keys()[0]]['err_mean']) + 1)
    for i, model in enumerate(models):
        # plot mean and std
        plt.plot(x, err_dict[model]['err_mean'], c=colors[i], label=model)
        plt.plot(x, err_dict[model]['err_mean'] + err_dict[model]['err_std'], '--', c=colors[i], alpha=0.1)
        plt.plot(x, err_dict[model]['err_mean'] - err_dict[model]['err_std'], '--', c=colors[i], alpha=0.1)
        plt.fill_between(x, err_dict[model]['err_mean'] - err_dict[model]['err_std'],
                         err_dict[model]['err_mean'] + err_dict[model]['err_std'], facecolor=colors[i], alpha=0.025, interpolate=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(err_dict) > 1:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim([0., plt.ylim()[1]])
    # plt.tight_layout()


def check_embed_match(X_embed1, X_embed2):
    """
    Check whether the two embeddings are almost the same by computing their normalized euclidean distances
    in the embedding space and checking the correlation.
    Inputs:
        - X_embed1, X_embed2: two Nxd matrices with coordinates in the embedding space
    Returns:
        - r: Pearson correlation coefficient between the normalized distances of the points
    """
    D_emb1 = pdist(X_embed1, 'euclidean')
    D_emb2 = pdist(X_embed2, 'euclidean')
    D_emb1 /= D_emb1.max()
    D_emb2 /= D_emb2.max()
    return np.corrcoef(D_emb1, D_emb2)[0, 1]
