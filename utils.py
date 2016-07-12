import colorsys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification, make_swiss_roll, make_s_curve
from sklearn.utils import check_random_state

def get_colors(N=100):
    HSV_tuples = [(x*1.0/N, 1., 0.8) for x in range(N)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

def center_K(K):
    # center the kernel matrix
    n, m = K.shape
    H = np.eye(n) - np.tile(1./n,(n,n))
    B = np.dot(np.dot(H,K),H)
    return (B+B.T)/2

def make_3_circles(n_samples, random_state=1):
    X = np.ones((3*n_samples, 3))
    Y_plot = np.ones((3*n_samples, 1))
    X[:n_samples,:2], _ = make_circles(n_samples=n_samples, noise=0.05, factor=.01, random_state=random_state)
    X[:n_samples,2] *= -1
    Y_plot[:n_samples,0] = 1
    X[n_samples:2*n_samples,:2], _ = make_circles(n_samples=n_samples, noise=0.05, factor=.01, random_state=random_state)
    X[n_samples:2*n_samples,2] = 0
    Y_plot[n_samples:2*n_samples,0] = 2
    X[2*n_samples:,:2], _ = make_circles(n_samples=n_samples, noise=0.05, factor=.01, random_state=random_state)
    Y_plot[2*n_samples:,0] = 3
    # shuffle examples
    idx = np.random.permutation(range(3*n_samples))
    X, Y_plot = X[idx,:], Y_plot[idx,:]
    # cut to actual size
    X, Y_plot = X[:n_samples,:], Y_plot[:n_samples,:]
    return X, Y_plot

def make_sphere(n_samples, random_state=1):
    # Create our sphere.
    random_state = check_random_state(random_state)
    p = random_state.rand(n_samples) * (2 * np.pi - 0.5)
    t = random_state.rand(n_samples) * np.pi

    # Sever the poles from the sphere.
    indices = ((t < (np.pi - (np.pi / 10))) & (t > ((np.pi / 10))))
    colors = p[indices]
    x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
        np.sin(t[indices]) * np.sin(p[indices]), \
        np.cos(t[indices])
    sphere_data = np.array([x, y, z]).T
    return sphere_data, colors

def make_broken_swiss_roll(n_samples, random_state=1):
    # get original swiss roll
    X, Y_plot = make_swiss_roll(2*n_samples, random_state=random_state)
    # cut off a part
    X, Y_plot = X[X[:,0]>-5,:], Y_plot[X[:,0]>-5]
    # get desired number of samples
    X, Y_plot = X[:n_samples,:], Y_plot[:n_samples]
    return X, Y_plot

def make_peaks(n_samples, random_state=1):
    # get randomly sampled 2d grid
    random_state = check_random_state(random_state)
    X = 10.*random_state.rand(n_samples, 3)
    # have as 3rd dimension some peaks
    X[X[:,0]<=5, 2] = np.cos(0.9*(X[X[:,0]<=5, 1]-2))
    X[X[:,0]>5, 2] = np.cos(0.5*(X[X[:,0]>5, 1]-5))
    # 3rd dim is also the color
    Y_plot = X[:,2]
    return X, Y_plot


def load_dataset(dataset, n_samples, random_state=1):
    # wrapper function to load one of the 3d datasets
    if dataset == 's_curve':
        return make_s_curve(n_samples, random_state=random_state)
    elif dataset == 'swiss_roll':
        return make_swiss_roll(n_samples, random_state=random_state)
    elif dataset == 'broken_swiss_roll':
        return make_broken_swiss_roll(n_samples, random_state=random_state)
    elif dataset == 'sphere':
        return make_sphere(n_samples, random_state=random_state)
    elif dataset == '3_circles':
        return make_3_circles(n_samples, random_state=random_state)
    elif dataset == 'peaks':
        return make_peaks(n_samples, random_state=random_state)
    else:
        print("unknown dataset")


def plot2d(X, Y_plot, X_test=None, Y_plot_test=None, title='original'):
    plt.figure()
    if (X_test is not None) and (Y_plot_test is not None):
        plt.scatter(X[:, 0], X[:, 1], c=Y_plot, alpha=0.2)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_plot_test, alpha=1)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=Y_plot, alpha=1)
    plt.title(title, fontsize=20)

def plot3d(X, Y_plot, X_test=None, Y_plot_test=None, title='original'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if (X_test is not None) and (Y_plot_test is not None):
        ax.scatter(X[:,0], X[:,1], X[:,2], c=Y_plot, alpha=0.2)
        ax.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=Y_plot_test, alpha=1)
    else:
        ax.scatter(X[:,0], X[:,1], X[:,2], c=Y_plot, alpha=1)
    plt.title(title, fontsize=20)


def plot_digits(X, digits, title=None):
    colorlist = get_colors(10)
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=colorlist[digits.target[i]],
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title, fontsize=20)
