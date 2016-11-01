from copy import deepcopy
import numpy as np
import theano
import theano.tensor as T
from theano import sparse

from ann import ANN


def thr_sigmoid(x):
    # apply a scaled version of the sigmoid to the input to map the data
    # between 0 and 1 if it's between 0 and 1 and threshold it otherwise
    return T.nnet.sigmoid(10. * (x - 0.5))


def shift_sigmoid(x):
    # apply a scaled version of the sigmoid to the input to map the data
    # between 0 and 1 if it's between 0 and 10 and threshold it otherwise
    return T.nnet.sigmoid(x - 5.)


def center_K(K):
    # center the kernel matrix
    n, m = K.shape
    H = np.eye(n) - np.tile(1. / n, (n, n))
    B = np.dot(np.dot(H, K), H)
    return (B + B.T) / 2


def embedding_error(s_est, s_true, error_fun, idx=None):
    if error_fun == 'squared':
        if idx:
            return T.mean((s_true[idx.nonzero()] - s_est[idx.nonzero()])**2)
        else:
            return T.mean((s_true - s_est)**2)
    elif error_fun == 'absolute':
        return T.mean(abs(s_true - s_est))
    else:
        # is what we were given a function in itself i.e. can we call it?
        try:
            return error_fun(s_est, s_true)
        except:
            raise Exception('Error function %s not implemented!' % error_fun)


class SimilarityEncoder(object):

    def __init__(self, n_targets, n_features, e_dim=2, n_out=[], activations=[None, None], error_fun='squared', sparse_features=False,
                 subsampling=False, lrate=0.1, lrate_decay=0.95, min_lrate=0.04, L1_reg=0., L2_reg=0., orthOT_reg=0.1, orthNN_reg=0., seed=12):
        """
        Constructs the Similarity Encoder
        by default it's a linear SimEc which can be used for visualization (i.e. output is 2D) and should give the same results as PCA/linear kPCA

        Inputs:
            - n_targets: for how many data points we know the similarities (typically X.shape[0], i.e. all training examples)
            - n_features: how many dimensions the original data has (i.e. X.shape[1])
            - e_dim: how many dimensions the embedding should have (default 2)
            - n_out: number of hidden units for other layers (last two are always fixed as e_dim and n_targets)
            - activations: for the NN model architecture
            - error_fun: which error measure should be used in backpropagation (default: 'squared', other values: 'absolute')
            - sparse_features: bool, whether the input features will be in form of a sparse matrix (csr)
            - lrate: learning rate (default 0.2)
            - lrate_decay: learning rate decay (default 0.95, set to 1 for no decay)
            - min_lrate: in case of lrate_decay, minimum to which the learning rate will decay (default 0.04)
            - L1_reg, L2_reg: standard NN weight regularization terms (default 0.)
            - orthOT_reg: regularization parameter to encourage orthogonal weights in the output layer (default 0.1)
                          (helpful to get the same solution as kPCA as there the embeddings are orthogonal as well (eigenvectors...))
            - orthNN_reg: regularization parameter to encourage orthogonal weights in the other layers besides the output layer (default 0.)
                          (this does not help in most cases, only for the linear SimEc to mimic regular PCA where the projection vectors are orthogonal)
            - seed: random seed for the NN initialization
        """
        # build the model
        self.n_targets = n_targets
        self.n_features = n_features
        # some parameters
        self.error_fun = error_fun
        self.learning_rate = theano.shared(
            value=lrate,
            name='learning_rate',
            borrow=True
        )
        self.lrate_decay = lrate_decay
        self.min_lrate = min_lrate

        # allocate symbolic variables for the data
        if sparse_features:
            self.x = sparse.csr_matrix('x')
        else:
            self.x = T.matrix('x')  # input data
        self.s = T.matrix('s')      # corresponding similarities
        # are we doing subsampling of the target similarities?
        if subsampling:
            idx = T.matrix('idx', dtype='int8')
        else:
            idx = None

        # construct the ANN
        self.model = ANN(
            x_in=self.x,
            n_in=n_features,
            n_out=n_out + [e_dim, n_targets],
            activation=activations,
            seed=seed
        )

        # the cost we minimize during training is the mean squared error of
        # the model plus the regularization terms (L1 and L2)
        if subsampling:
            self.cost = (
                embedding_error(self.model.output, self.s, self.error_fun, idx)
                + L1_reg * self.model.L1
                + L2_reg * self.model.L2_sqr
                + orthNN_reg * self.model.orthNN
                + orthOT_reg * self.model.orthOT
            )
        else:
            self.cost = (
                embedding_error(self.model.output, self.s, self.error_fun)
                + L1_reg * self.model.L1
                + L2_reg * self.model.L2_sqr
                + orthNN_reg * self.model.orthNN
                + orthOT_reg * self.model.orthOT
            )

        # compile a Theano function that computes the embedding on some data
        self.embed = theano.function(
            inputs=[self.x],
            outputs=self.model.layers[-2].output
        )

        # compute the gradient of cost with respect to all parameters
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(self.cost, param) for param in self.model.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(self.model.params, gparams)
        ]

        # compile a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        if subsampling:
            self.train_model = theano.function(
                inputs=[self.x, self.s, idx],
                outputs=embedding_error(self.model.output, self.s, self.error_fun, idx),
                updates=updates,
                allow_input_downcast=True
            )
        else:
            self.train_model = theano.function(
                inputs=[self.x, self.s],
                outputs=embedding_error(self.model.output, self.s, self.error_fun),
                updates=updates,
                allow_input_downcast=True
            )
        # to only get the error
        self.test_model = theano.function(
            inputs=[self.x, self.s],
            outputs=embedding_error(self.model.output, self.s, self.error_fun)
        )

    def fit(self, X, S, idx=None, verbose=True, max_epochs=5000):
        """
        fit the model on some training data

        Inputs:
            - X: training data (n_train x n_features)
            - S: target similarities for all the training points (n_train x n_targets)
            - idx: if subsampling is on, this can be a matrix of 0 and 1 the same shape as S
                   to indicate which of the similarities should be used for learning
            - verbose: bool, whether to output state of training
            - max_epochs: max number of times to go through the training data (default 5000)
        """
        assert X.shape[0] == S.shape[0], "need target similarities for all training examples"
        assert X.shape[1] == self.n_features, "wrong number of features specified when initializing the model"
        assert S.shape[1] == self.n_targets, "wrong number of targets specified when initializing the model"
        # normalize similarity matrix, other wise the weights will overshoot (turn
        # to nan) / we would have to be too careful with the learning rate
        S /= np.max(np.abs(S))
        # define some variables for training
        n_train = X.shape[0]
        # work on 20 training examples at a time
        batch_size = min(n_train, 100)
        n_batches = int(np.ceil(float(n_train) / batch_size))

        # do the actual training of the model
        best_error = np.inf
        best_layers = [deepcopy(p) for p in self.model.layers]
        mean_train_error = []
        for e in range(max_epochs):
            if verbose:
                if not e or not (e + 1) % 25:
                    print("Epoch %i" % (e + 1))
            train_error = []
            for bi in range(n_batches):
                mini_s = S[bi * batch_size:min((bi + 1) * batch_size, n_train), :]
                mini_x = X[bi * batch_size:min((bi + 1) * batch_size, n_train), :]
                # train model
                if idx is not None:
                    mini_idx = idx[bi * batch_size:min((bi + 1) * batch_size, n_train), :]
                    train_error.append(self.train_model(mini_x, mini_s, mini_idx))
                else:
                    train_error.append(self.train_model(mini_x, mini_s))
            mean_train_error.append(np.mean(train_error))
            if not e or not (e + 1) % 25:
                if verbose:
                    print("Mean training error: %.10f" % mean_train_error[-1])
                # adapt learning rate
                if e > 300 and (mean_train_error[-1] - 0.001 > best_error):
                    # we're bouncing, the learning rate is WAY TO HIGH
                    self.learning_rate.set_value(self.learning_rate.get_value() * 0.75)
                    # might be a problem of min_lrate as well
                    if self.learning_rate.get_value() < self.min_lrate:
                        self.min_lrate *= 0.7
                    print("Learning rate too high! Reseting to best local minima (error: %.10f)." % best_error)
                    for i, l in enumerate(best_layers):
                        self.model.layers[i].W.set_value(l.W.get_value(borrow=False))
                        self.model.layers[i].b.set_value(l.b.get_value(borrow=False))
                    test_error = []
                    for bi in range(n_batches):
                        mini_s = S[bi * batch_size:min((bi + 1) * batch_size, n_train), :]
                        mini_x = X[bi * batch_size:min((bi + 1) * batch_size, n_train), :]
                        # test model
                        test_error.append(self.test_model(mini_x, mini_s))
                    print("Sanity check, mean test error: %.10f" % np.mean(test_error))
                else:
                    self.learning_rate.set_value(max(self.min_lrate, self.learning_rate.get_value() * self.lrate_decay))
            # store best model
            if mean_train_error[-1] < best_error:
                best_error = mean_train_error[-1]
                best_layers = [deepcopy(p) for p in self.model.layers]
            # converged?
            if e > 500 and (mean_train_error[-25] - mean_train_error[-1] <= 0.00000001):
                break
        # use the best model
        for i, l in enumerate(best_layers):
            self.model.layers[i].W.set_value(l.W.get_value(borrow=False))
            self.model.layers[i].b.set_value(l.b.get_value(borrow=False))
        print("Final training error: %.10f; lowest error: %.10f" % (mean_train_error[-1], best_error))
        # one last time just to get the error to double check
        test_error = []
        for bi in range(n_batches):
            mini_s = S[bi * batch_size:min((bi + 1) * batch_size, n_train), :]
            mini_x = X[bi * batch_size:min((bi + 1) * batch_size, n_train), :]
            # test model
            test_error.append(self.test_model(mini_x, mini_s))
        print("Last mean error: %.10f" % np.mean(test_error))

    def transform(self, X):
        """
        using a fitted model, embed the given data

        Inputs:
            - X: some data with the same features as the original training data (i.e. n x n_features)
                 if the original features were sparse, these have to sparse as well

        Returns:
            - X_embed: the embedded data points (n x e_dim)
        """
        assert X.shape[1] == self.n_features, "number of features doesn't match model architecture"
        return self.embed(X)
