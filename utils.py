from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr


def get_colors(N=100):
    HSV_tuples = [(x * 1. / (N+1), 1., 0.8) for x in range(N)]
    return [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]


def plot_mnist(X, y, X_test=None, y_test=None, title=None):
    plt.figure()
    colorlist = get_colors(10)
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    if (X_test is not None) and (y_test is not None):
        x_min, x_max = np.min(np.array([x_min, np.min(X_test, 0)]), 0), np.max(np.array([x_max, np.max(X_test, 0)]), 0)
        X_test = (X_test - x_min) / (x_max - x_min)
    X = (X - x_min) / (x_max - x_min)
    if (X_test is not None) and (y_test is not None):
        for i in range(X_test.shape[0]):
            plt.text(X_test[i, 0], X_test[i, 1], str(y_test[i]),
                     color=colorlist[y_test[i]],
                     fontdict={'weight': 'medium', 'size': 'smaller'},
                     alpha=0.4)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colorlist[y[i]],
                 fontdict={'weight': 'medium', 'size': 'smaller'},
                 alpha=1.)
    plt.xticks([]), plt.yticks([])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    if title is not None:
        plt.title(title)


def plot_20news(X, y, target_names, X_test=None, y_test=None, title=None, legend=False):
    colorlist = get_colors(len(target_names))

    def plot_scatter(X, y, alpha=1):
        y = np.array(y)
        for i, l in enumerate(target_names):
            plt.scatter(X[y == i, 0], X[y == i, 1], c=colorlist[i], alpha=alpha,
                        edgecolors='none', label=l if alpha >= 0.5 else None)  # , rasterized=True)
    # plot scatter plot
    plt.figure()
    if (X_test is not None) and (y_test is not None):
        plot_scatter(X_test, y_test, 0.4)
        plot_scatter(X, y, 1.)
    else:
        plot_scatter(X, y, 0.6)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def check_embed_match(X_embed1, X_embed2):
    """
    Check whether the two embeddings are almost the same by computing their normalized euclidean distances
    in the embedding space and checking the correlation.
    Inputs:
        - X_embed1, X_embed2: two Nxd matrices with coordinates in the embedding space
    Returns:
        - msq, r^2, rho: mean squared error, R^2, and Spearman correlation coefficient between the distance matrices of
                         both embeddings (mean squared error is more exact, corrcoef a more relaxed error measure)
    """
    D_emb1 = pdist(X_embed1, 'euclidean')
    D_emb2 = pdist(X_embed2, 'euclidean')
    D_emb1 /= D_emb1.max()
    D_emb2 /= D_emb2.max()
    # compute mean squared error
    msqe = np.mean((D_emb1 - D_emb2) ** 2)
    # compute Spearman correlation coefficient
    rho = spearmanr(D_emb1.flatten(), D_emb2.flatten())[0]
    # compute Pearson correlation coefficient
    r = pearsonr(D_emb1.flatten(), D_emb2.flatten())[0]
    return msqe, r**2, rho


def check_similarity_match(X_embed, S, X_embed_is_S_approx=False, norm=False):
    """
    Since SimEcs are supposed to project the data into an embedding space where the target similarities
    can be linearly approximated; check if X_embed*X_embed^T = S
    (check mean squared error, R^2, and Spearman correlation coefficient)
    Inputs:
        - X_embed: Nxd matrix with coordinates in the embedding space
        - S: NxN matrix with target similarities (do whatever transformations were done before using this
             as input to the SimEc, e.g. centering, etc.)
    Returns:
        - msq, r^2, rho: mean squared error, R^2, and Spearman correlation coefficient between linear kernel of embedding
                         and target similarities (mean squared error is more exact, corrcoef a more relaxed error measure)
    """
    if X_embed_is_S_approx:
        S_approx = X_embed
    else:
        # compute linear kernel as approximated similarities
        S_approx = X_embed.dot(X_embed.T).real
    # to get results that are comparable across similarity measures, we have to normalize them somehow,
    # in this case by dividing by the absolute max value of the target similarity matrix
    if norm:
        S_norm = S / np.max(np.abs(S))
        S_approx /= np.max(np.abs(S_approx))
    else:
        S_norm = S
    # compute mean squared error
    msqe = np.mean((S_norm - S_approx) ** 2)
    # compute Spearman correlation coefficient
    rho = spearmanr(S_norm.flatten(), S_approx.flatten())[0]
    # compute Pearson correlation coefficient
    r = pearsonr(S_norm.flatten(), S_approx.flatten())[0]
    return msqe, r**2, rho
