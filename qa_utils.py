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
    iu = np.triu_indices(S.shape[0], 1)
    sims = S[iu]
    dists = D[iu]
    # sort by closest distances and highest similarities (globally, not for individual data points!)
    NN_O = np.argsort(sims)[::-1]
    NN_E = np.argsort(dists)
    # compute raw error for every NN position
    E = dists[NN_O] - dists[NN_E]
    # points too close don't count, thats a problem of the embedding error
    # (but don't just set them to 0 or it'll ruin the mean)
    E[E < 0] = 0.
    # bin by NN
    parts = np.linspace(0, len(E), 101, dtype=int)
    sims = sims[NN_O]
    # buffer beginning and end
    s = [sims[0]]
    err_mean, err_std = [E[0] if E[0] >= 0 else 0.], [0.]
    for i, p in enumerate(parts[:-1]):
        val = E[p:parts[i+1]]
        val = val[val >= 0]
        err_mean.append(np.mean(val) if len(val) else 0.)
        err_std.append(np.std(val) if len(val) else 0.)
        s.append(np.mean(sims[p:parts[i+1]]))
    err_mean.append(E[-1] if E[-1] >= 0 else 0.)
    err_std.append(0.)
    s.append(sims[-1])
    return np.array(err_mean), np.array(err_std), np.array(s)


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
    iu = np.triu_indices(S.shape[0], 1)
    sims = S[iu]
    dists = D[iu]
    # sort by closest distances and highest similarities (globally, not for individual data points!)
    NN_O = np.argsort(sims)[::-1]
    NN_E = np.argsort(dists)
    # compute raw error for every NN position
    E = sims[NN_O] - sims[NN_E]
    # "too" similar is ok - the original error deals with this
    # (but don't just set them to 0 or it'll ruin the mean)
    E[E < 0] = 0.
    # bin by embedding distances
    parts = np.linspace(0, len(E), 101, dtype=int)
    dists = dists[NN_E]
    d = [dists[0]]
    err_mean, err_std = [E[0] if E[0] >= 0 else 0.], [0.]
    for i, p in enumerate(parts[:-1]):
        val = E[p:parts[i+1]]
        val = val[val >= 0]
        err_mean.append(np.mean(val) if len(val) else 0.)
        err_std.append(np.std(val) if len(val) else 0.)
        d.append(np.mean(dists[p:parts[i+1]]))
    err_mean.append(E[-1] if E[-1] >= 0 else 0.)
    err_std.append(0.)
    d.append(dists[-1])
    return np.array(err_mean), np.array(err_std), np.array(d)


def plot_errors(err_dict, title, xlabel='', ylabel='error', xlim=[0., 1.], ylim=None, models=[]):
    """
    Inputs:
        - err_dict: a dict with {'model':{'err_mean':[], 'err_min':[], 'err_max':[]}} with mean, min and max error values for multiple models
    """
    plt.figure()
    if not models:
        models = sorted(err_dict.keys())
    colors = get_colors(len(models))
    for i, model in enumerate(models):
        # plot mean and std
        x = err_dict[model]['x']  # corresponding distances or similarities
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
