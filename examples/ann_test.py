from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import theano.tensor as T
from sklearn.datasets import make_moons, make_circles, make_classification

from simec.ann_models import SupervisedNNModel


def linear_regression(y_dim=5, x_dim=10):
    # generate training and test data
    np.random.seed(15)
    n_train = 1000
    W = np.random.randint(-4, 5, size=(x_dim, y_dim))
    b = np.random.randint(-2, 2, size=(1, y_dim))
    X = np.random.rand(n_train, x_dim)
    Y = np.dot(X, W) + b
    X += np.random.randn(n_train, x_dim) * 0.02
    X_test = np.random.rand(20, x_dim)
    Y_test = np.dot(X_test, W) + b
    X_test += np.random.randn(20, x_dim) * 0.02

    # build, train, and test the model
    model = SupervisedNNModel(x_dim, y_dim)
    model.fit(X, Y)
    print("Test Error: %f" % model.score(X_test, Y_test))

    if x_dim == 1 and y_dim == 1:
        plt.figure()
        plt.plot(X[:, 0], Y[:, 0], 'm*', label='data samples')
        X_plot = np.linspace(np.min(X), np.max(X), 1000)
        plt.plot(X_plot, model.predict(X_plot[:, np.newaxis]), 'g', label='prediction')
        plt.plot(X_plot, np.dot(X_plot[:, np.newaxis], W) + b, 'k', label='true curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression Problem')
        plt.legend()


def nonlinear_regression(y_dim=1, x_dim=1):
    # generate training and test data
    np.random.seed(15)
    n_train = 1000
    X = np.random.rand(n_train, x_dim) * np.pi * 2.
    Y = np.sin(X)
    X += np.random.randn(n_train, x_dim) * 0.02
    X_test = np.random.rand(20, x_dim) * np.pi * 2.
    Y_test = np.sin(X_test)
    X_test += np.random.randn(20, x_dim) * 0.02

    # build, train, and test the model
    model = SupervisedNNModel(x_dim, y_dim, hunits=[100, 50], activations=[T.tanh, T.tanh, None])
    model.fit(X, Y)
    print("Test Error: %f" % model.score(X_test, Y_test))

    if x_dim == 1 and y_dim == 1:
        plt.figure()
        plt.plot(X[:, 0], Y[:, 0], 'm*', label='data samples')
        X_plot = np.linspace(np.min(X), np.max(X), 1000)
        plt.plot(X_plot, model.predict(X_plot[:, np.newaxis]), 'g', label='prediction')
        plt.plot(X_plot, np.sin(X_plot), 'k', label='true curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Non-Linear Regression Problem')
        plt.legend()


def classification(dataset=0):
    # generate training and test data
    n_train = 1000
    if dataset == 0:
        X, Y = make_classification(n_samples=n_train, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        X_test, Y_test = make_classification(n_samples=50, n_features=2, n_redundant=0, n_informative=2,
                                             random_state=1, n_clusters_per_class=1)
        X_test += 2 * rng.uniform(size=X_test.shape)
    elif dataset == 1:
        X, Y = make_moons(n_samples=n_train, noise=0.3, random_state=0)
        X_test, Y_test = make_moons(n_samples=50, noise=0.3, random_state=1)
    elif dataset == 2:
        X, Y = make_circles(n_samples=n_train, noise=0.2, factor=0.5, random_state=1)
        X_test, Y_test = make_circles(n_samples=50, noise=0.2, factor=0.5, random_state=1)
    else:
        print("dataset unknown")
        return

    # build, train, and test the model
    model = SupervisedNNModel(X.shape[1], 2, hunits=[100, 50], activations=[T.tanh, T.tanh, T.nnet.softmax], cost_fun='negative_log_likelihood',
                              error_fun='zero_one_loss', learning_rate=0.01, L1_reg=0., L2_reg=0.)
    model.fit(X, Y)
    print("Test Error: %f" % model.score(X_test, Y_test))

    # plot dataset + predictions
    plt.figure()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright, alpha=0.6)
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm_bright)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Classification Problem (%i)' % dataset)


if __name__ == '__main__':
    linear_regression(1, 1)
    nonlinear_regression(1, 1)
    classification(0)
    classification(1)
    classification(2)
    plt.show()
