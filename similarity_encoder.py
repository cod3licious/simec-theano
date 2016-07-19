from copy import deepcopy
import numpy as np
import theano
import theano.tensor as T
from theano import sparse

from ann import ANN

def thr_sigmoid(x):
    # apply a scaled version of the sigmoid to the input to map the data 
    # between 0 and 1 if it's between 0 and 1 and threshold it otherwise
    return T.nnet.sigmoid(10.*(x-0.5))

def embedding_error(s_est, s_true, error_fun):
    if error_fun == 'squared':
        return T.mean((s_true-s_est)**2)
    elif error_fun == 'abs':
        return T.mean(abs(s_true-s_est))
    else:
        raise Exception('Error function %s not implemented!' % error_fun)

class SimilarityEncoder(object):

    def __init__(self, n_targets, n_features, e_dim=2, n_out=[], activations=[None, None], error_fun='squared', seed=12, sparse_features=False, 
                 lrate=0.1, lrate_decay=0.95, min_lrate=0.04, L1_reg=0., L2_reg=0., orthOT_reg=0.1, orthNN_reg=0.):
        """
        Constructs the Similarity Encoder

        Inputs:
            - n_targets: for how many data points we know the similarities (typically X.shape[0], i.e. all training examples)
            - n_features: how many dimensions the original data has (i.e. X.shape[1])
            - e_dim: how many dimensions the embedding should have (default 2)
            - n_out: number of hidden units for other layers (last two are always fixed as e_dim and n_targets)
            - activations: for the NN model architecture
            - error_fun: which error measure should be used in backpropagation (default: 'squared', other values: 'abs')
            - seed: random seed for the NN initialization
            - sparse_features: bool, whether the input features will be in form of a sparse matrix (csr)
            - lrate: learning rate (default 0.2)
            - lrate_decay: learning rate decay (default 0.95, set to 1 for no decay)
            - min_lrate: in case of lrate_decay, minimum to which the learning rate will decay (default 0.04)
            - L1_reg, L2_reg: standard NN weight regularization terms (default 0.)
            - orthOT_reg: regularization parameter to encourage orthogonal weights in the output layer (default 0.1)
                          (helpful to get the same solution as kPCA as there the embeddings are orthogonal as well (eigenvectors...))
            - orthNN_reg: regularization parameter to encourage orthogonal weights in the other layers besides the output layer (default 0.)
                          (this does not help in most cases, only for the linear SimEc to mimic regular PCA where the projection vectors are orthogonal)
        """
        ## build the model
        self.n_targets = n_targets
        self.n_features = n_features
        # some parameters
        self.error_fun = error_fun
        self.learning_rate = lrate
        self.lrate_decay = lrate_decay
        self.min_lrate = min_lrate

        # allocate symbolic variables for the data
        if sparse_features:
            x = sparse.csr_matrix('x')
        else:
            x = T.matrix('x')  # input data
        s = T.matrix('s')      # corresponding similarities

        # construct the ANN
        self.model = ANN(
            x_in=x,
            n_in=n_features,
            n_out=n_out+[e_dim, n_targets],
            activation=activations,
            seed=seed
        )

        # the cost we minimize during training is the mean squared error of
        # the model plus the regularization terms (L1 and L2)
        self.cost = (
            embedding_error(self.model.output, s, self.error_fun)
            + L1_reg * self.model.L1
            + L2_reg * self.model.L2_sqr
            + orthNN_reg * self.model.orthNN
            + orthOT_reg * self.model.orthOT
        )

        # compile a Theano function that computes the embedding on some data
        self.embed = theano.function(
            inputs=[x],
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
            inputs=[x, s],
            outputs=embedding_error(self.model.output, s, self.error_fun),
            updates=updates
        )

    def fit(self, X, S, verbose=True):
        """
        fit the model on some training data

        Inputs:
            - X: training data (n_train x n_features)
            - S: target similarities for all the training points (n_train x n_targets)
            - verbose: bool, whether to output state of training
        """
        assert X.shape[0] == S.shape[0], "need target similarities for all training examples"
        assert X.shape[1] == self.n_features, "wrong number of features specified when initializing the model"
        assert S.shape[1] == self.n_targets, "wrong number of targets specified when initializing the model"
        # normalize similarity matrix, other wise the weights will overshoot (turn to nan) / we would have to be too careful with the learning rate
        S /= np.max(np.abs(S))
        ## define some variables for training
        n_train = X.shape[0]
        # number of times to go through the training data
        max_epochs = 5000
        # work on 20 training examples at a time
        batch_size = min(n_train, 100)
        n_batches = int(np.ceil(float(n_train)/batch_size))
        
        ## do the actual training of the model
        best_error = np.inf
        best_model = deepcopy(self.model)
        mean_train_error = []
        for e in range(max_epochs):
            if verbose:
                if not e or not (e+1) % 25:
                    print("Epoch %i" % (e+1))
            train_error = []
            for bi in range(n_batches):
                mini_s = S[bi*batch_size:min((bi+1)*batch_size,n_train),:]
                mini_x = X[bi*batch_size:min((bi+1)*batch_size,n_train),:]
                # train model
                train_error.append(self.train_model(mini_x, mini_s))
            mean_train_error.append(np.mean(train_error))
            if not e or not (e+1) % 25:
                if verbose:
                    print("Mean training error: %.10f" % mean_train_error[-1])
                # store best model
                if mean_train_error[-1] < best_error:
                    best_error = mean_train_error[-1]
                    best_model = deepcopy(self.model)
                # adapt learning rate
                self.learning_rate = max(self.min_lrate, self.learning_rate*self.lrate_decay)
            if e > 500 and (mean_train_error[-1]-0.05 > mean_train_error[-20] or round(mean_train_error[-1], 10) == round(mean_train_error[-5], 10)):
                break
        # use the best model
        self.model = best_model
        print("Final training error: %f; lowest error: %f" % (mean_train_error[-1], best_error))

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
