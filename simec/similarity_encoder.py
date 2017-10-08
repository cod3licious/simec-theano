from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range
from builtins import object
from copy import deepcopy
import numpy as np
import theano
import theano.tensor as T
from theano import sparse

from .ann import ANN


def scaled_sigmoid(x):
    # apply a scaled version of the sigmoid to the input to map the data
    # between 0 and 1 if it's between 0 and 1 and threshold it otherwise
    return T.nnet.sigmoid(10. * (x - 0.5))


def scaled_sigmoid_nonzero(x):
    # apply a scaled version of the sigmoid to the input to map the data
    # steep rise
    m = 8.
    # is 0.5 for x0 (if c = 0)
    x0 = x.min() + 0.5*(x.max()-x.min())
    # offset of c, i.e. sig \in [c, 1]
    c = 0.
    return (1.-c)*T.nnet.sigmoid(m * (x - x0)) + c


def exp_cutoff(x):
    # 1 or less with heavy tail
    # the upper 80% of the values is at 1
    b = 0.8
    c = x.min() + b*(x.max()-x.min())
    # the lower 20% of the values is below y
    y = 0.15
    a = y/T.exp((0.2-b)*(x.max()-x.min()))
    return T.minimum(a * T.exp(x - c), 1.)


def minmax_norm(x):
    # normalize the scores to be between c and 1
    c = 0.00000000000000000001
    return (1.-c)*(x-x.min())/x.max() + c


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
    if s_true is None:
        # for s_ll_reg without a matrix
        return 0.
    if error_fun == 'squared':
        if idx is not None:
            return T.mean((s_true[idx.nonzero()] - s_est[idx.nonzero()])**2)
        else:
            return T.mean((s_true - s_est)**2)
    elif error_fun == 'absolute':
        return T.mean(abs(s_true - s_est))
    elif error_fun == 'cross-entropy':
        return -T.mean(s_true * T.log(s_est) + (1 - s_true) * T.log(1 - s_est))
    elif error_fun == 'kl-divergence':
        # make sure s_est is never truly 0 otherwise you'll get errors (this is your responsibility!)
        # also since log(0) is not defined but the kl-divergence is 0 if s_true is 0, we can ignore these values
        return T.mean(abs(s_true * T.log(s_true/s_est)))
    elif error_fun == 'symkl-divergence':
        return T.mean(s_true * T.log(s_true/s_est) + s_est * T.log(s_est/s_true))
    else:
        # is what we were given a function in itself i.e. can we call it?
        try:
            return error_fun(s_est, s_true)
        except:
            raise Exception('Error function %s not implemented!' % error_fun)


class SimilarityEncoder(object):

    def __init__(self, n_targets, n_features, e_dim=2, n_out=[], activations=[None, None], error_fun='squared', sparse_features=False,
                 subsampling=False, lrate=0.1, lrate_control=300, lrate_decay=1., min_lrate=0., s_ll_reg=0., L1_reg=0., L2_reg=0., L2_last_reg=0.,
                 orthOT_reg=0., orthNN_reg=0., normOT_reg=0., seed=12):
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
            - lrate: learning rate (default 0.1)
            - lrate_control: number of epochs after which, if the training error is higher than in a previous iteration,
                             the learning rate should be decreased.
                             this helps to find a minimum if the lrate was chosen too large. (default 300)
            - lrate_decay: learning rate decay (default 1., set to less than 1 to get a decay)
            - min_lrate: in case of lrate_decay, minimum to which the learning rate will decay (default 0.)
            - s_ll_reg: encourage the dot product of the last layer to approximate the target similarities as well (recommended, default 0.)
                        if this is > 0 then fit() needs S_ll as an argument
            - L1_reg, L2_reg: standard NN weight regularization terms (default 0.)
            - L2_reg_last: L2 regularization of only the last layer, mapping embedding to targets (default 0.)
            - orthOT_reg: regularization parameter to encourage orthogonal weights in the output layer (default 0.)
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
        self.lrate_control = lrate_control
        self.lrate_decay = lrate_decay
        self.min_lrate = min_lrate

        # allocate symbolic variables for the data
        if sparse_features:
            self.x = sparse.csr_matrix('x')
        else:
            self.x = T.matrix('x')  # input data
        self.s = T.matrix('s')      # corresponding similarities

        # construct the ANN
        self.model = ANN(
            x_in=self.x,
            n_in=n_features,
            n_out=n_out + [e_dim, n_targets],
            activation=activations,
            seed=seed
        )

        # the cost we minimize during training is the mean squared error of
        # the model plus the regularization terms (L1, L2, etc.)
        self.cost = (L1_reg * self.model.L1
                     + L2_reg * self.model.L2_sqr
                     + L2_last_reg * self.model.L2_last
                     + orthNN_reg * self.model.orthNN
                     + orthOT_reg * self.model.orthOT
                     + normOT_reg * self.model.normOT)

        training_inputs = [self.x, self.s]
        # if we're getting an additional sim mat for the last layer
        if s_ll_reg:
            self.s_ll = T.matrix('s_ll')
            self.cost += s_ll_reg * embedding_error(self.model.s_approx_ll, self.s_ll, self.error_fun)
            training_inputs.append(self.s_ll)
        else:
            self.s_ll = None

        # are we doing subsampling of the target similarities?
        if subsampling:
            idx = T.matrix('idx', dtype='int8')
            self.cost += embedding_error(self.model.output, self.s, self.error_fun, idx)
            training_inputs.append(idx)
            training_outputs = embedding_error(self.model.output, self.s, self.error_fun, idx)
        else:
            idx = None
            self.cost += embedding_error(self.model.output, self.s, self.error_fun)
            training_outputs = embedding_error(self.model.output, self.s, self.error_fun)

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
        self.train_model = theano.function(
            inputs=training_inputs,
            outputs=training_outputs,
            updates=updates,
            allow_input_downcast=True
        )
        # to only get the error
        self.test_model = theano.function(
            inputs=[self.x, self.s],
            outputs=embedding_error(self.model.output, self.s, self.error_fun)
        )

    def fit(self, X, S, S_ll=None, idx=None, verbose=True, max_epochs=5000):
        """
        fit the model on some training data

        Inputs:
            - X: training data (n_train x n_features)
            - S: target similarities for all the training points (n_train x n_targets)
                 this should be centered and normalized as S/np.max(np.abs(S)) since otherwise
                 the weights might overshoot or we would have to be careful with the learning rate
            - S_ll: target similarities for the last layer (n_targets x n_targets) - if s_ll_reg > 0.
            - idx: if subsampling is on, this can be a matrix of 0 and 1 the same shape as S
                   to indicate which of the similarities should be used for learning
            - verbose: bool, whether to output state of training
            - max_epochs: max number of times to go through the training data (default 5000)
        """
        assert X.shape[0] == S.shape[0], "need target similarities for all training examples"
        assert X.shape[1] == self.n_features, "wrong number of features specified when initializing the model"
        assert S.shape[1] == self.n_targets, "wrong number of targets specified when initializing the model"
        # define some variables for training
        n_train = X.shape[0]
        # work on 100 training examples at a time
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
                    if S_ll is not None:
                        train_error.append(self.train_model(mini_x, mini_s, S_ll, mini_idx))
                    else:
                        train_error.append(self.train_model(mini_x, mini_s, mini_idx))
                elif S_ll is not None:
                    train_error.append(self.train_model(mini_x, mini_s, S_ll))
                else:
                    train_error.append(self.train_model(mini_x, mini_s))
            mean_train_error.append(np.mean(train_error))
            if not e or not (e + 1) % 25:
                if verbose:
                    print("Mean training error: %.10f" % mean_train_error[-1])
                # adapt learning rate
                if (e > self.lrate_control and (mean_train_error[-1] - 0.00001 > best_error)) or np.isnan(mean_train_error[-1]):
                    # we're bouncing, the learning rate is WAY TO HIGH
                    self.learning_rate.set_value(self.learning_rate.get_value() * 0.75)
                    # might be a problem of min_lrate as well
                    if self.learning_rate.get_value() < self.min_lrate:
                        self.min_lrate *= 0.7
                    if verbose:
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
                    if verbose:
                        print("Sanity check, mean test error: %.10f" % np.mean(test_error))
                else:
                    self.learning_rate.set_value(max(self.min_lrate, self.learning_rate.get_value() * self.lrate_decay))
            # store best model
            if mean_train_error[-1] < best_error:
                best_error = mean_train_error[-1]
                best_layers = [deepcopy(p) for p in self.model.layers]
            # converged?
            if e > 500 and (abs(mean_train_error[-50] - mean_train_error[-1]) <= 0.00000001):
                if verbose:
                    print("Converged, terminating early.")
                break
        # use the best model
        for i, l in enumerate(best_layers):
            self.model.layers[i].W.set_value(l.W.get_value(borrow=False))
            self.model.layers[i].b.set_value(l.b.get_value(borrow=False))
        print("Final training error after %i epochs: %.10f; lowest error: %.10f" % (e+1, mean_train_error[-1], best_error))
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
