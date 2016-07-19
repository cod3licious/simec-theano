import numpy as np
import theano
import theano.tensor as T

from ann import ANN


def error_wrapper(y_pred, y_true, error_fun):
    if error_fun == 'mean_squared_error':
        return mean_squared_error(y_pred, y_true)
    elif error_fun == 'negative_log_likelihood':
        return negative_log_likelihood(y_pred, y_true)
    elif error_fun == 'zero_one_loss':
        # get the actual labels for y_pred first
        return zero_one_loss(T.argmax(y_pred, axis=1), y_true)
    else:
        raise Exception('Error function %s not implemented!' % error_fun)


def mean_squared_error(y_pred, y_true):
    return T.mean((y_true-y_pred)**2)

def negative_log_likelihood(y_out, y):
    """Return the mean of the negative log-likelihood of the prediction
    of a model under a given target distribution.

    Inputs:
        - y_out: N x n_out theano matrix with class probabilities for all N data points
                 (predicted label = argmax(y_out, axis=1))
        - y: N dimensional vector with corresponding true class labels
             (in the range [0, n_out)) for all data points
    
    Returns:
        - scalar representing the mean negative log likelihood

    Note: we use the mean instead of the sum so that
          the learning rate is less dependent on the batch size
    """
    # y.shape[0] is (symbolically) the number of rows in y, i.e.,
    # number of examples (N) in the minibatch
    # T.arange(y.shape[0]) is a symbolic vector which will contain
    # [0,1,2,... n-1] T.log(y_out) is a matrix of
    # Log-Probabilities (call it LP) with one row per example and
    # one column per class LP[T.arange(y.shape[0]),y] is a vector
    # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
    # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
    # the mean (across minibatch examples) of the elements in v,
    # i.e., the mean log-likelihood across the minibatch.
    return -T.mean(T.log(y_out)[T.arange(y.shape[0]), y])

def zero_one_loss(y_pred, y_true):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch

    Inputs:
        - y_pred:  N dimensional vector containing the predicted classes
        - y_true: N dimensional vector containing the true class labels
    Returns:
        - zero-one-loss
    """
    # the T.neq operator returns a vector of 0s and 1s, where 1
    # represents a mistake in prediction
    return T.mean(T.neq(y_pred, y_true))


class SupervisedNNModel(object):

    def __init__(self, x_dim, y_dim, hunits=[], activations=[None], cost_fun='mean_squared_error', error_fun='mean_squared_error',
                 learning_rate=0.001, L1_reg=0.00, L2_reg=0.001):
        """
        initialize the model

        Inputs:
            - x_dim: dimensionality (number of features) of the input data
            - y_dim: dimensionality of the output (last layer)
            - hunits: number of hidden units in multiple hidden layers (needs to be a list corresponding to the activations)
            - activations: for additional hidden layers possible non-linear activation functions
                           (list, same length as hunits + 1 (for y_dim)), default [None]
            - cost_fun: type of error used in backpropagation to train the model (default 'mean_squared_error' for regression)
            - error_fun: type of error measure used to compute the test error (default 'mean_squared_error' for regression)

        Examples:

        Linear Regression:
            model = SupervisedNNModel(x_dim, y_dim, hunits=[], activations=[None])
        Non-Linear Regression:
            model = SupervisedNNModel(x_dim, y_dim, hunits=[100, 50], activations=[T.tanh, T.tanh, None])
        Classification:
            model = SupervisedNNModel(x_dim, y_dim, hunits=[100, 50], activations=[T.tanh, T.tanh, T.nnet.softmax],
                                      cost_fun='negative_log_likelihood', error_fun='zero_one_loss')
        """

        ## build the model
        # some parameters
        self.x_dim = x_dim
        self.y_dim = y_dim

        # allocate symbolic variables for the data
        x = T.matrix('x')    # input data
        if error_fun == 'mean_squared_error':
            y = T.matrix('y')    # corresponding labels
        else:
            y = T.ivector('y')   # for classification we have indices

        # construct the ANN
        self.model = ANN(
            x_in=x,
            n_in=x_dim,
            n_out=hunits+[y_dim],
            activation=activations,
            seed=12
        )

        # the cost we minimize during training is the mean squared error of
        # the model plus the regularization terms (L1 and L2)
        cost = (
            error_wrapper(self.model.output, y, cost_fun)
            + L1_reg * self.model.L1
            + L2_reg * self.model.L2_sqr
        )

        # compile a Theano function that computes the error on some test data
        self.test_model = theano.function(
            inputs=[x, y],
            outputs=error_wrapper(self.model.output, y, error_fun),
            allow_input_downcast=True
        )

        self.predict = theano.function(
            inputs=[x],
            outputs=self.model.output
        )

        # compute the gradient of cost with respect to all parameters
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in self.model.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.model.params, gparams)
        ]

        # compile a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(
            inputs=[x, y],
            outputs=error_wrapper(self.model.output, y, error_fun),
            updates=updates,
            allow_input_downcast=True
        )

    def fit(self, X, Y):
        """
        fit the model

        Inputs:
            - X: input data of dimensions N x x_dim
            - Y: labels corresponding to the input data, size N x y_dim
        """
        assert X.shape[0] == Y.shape[0], "need labels for all training examples"
        assert X.shape[1] == self.x_dim, "wrong number of features specified when initializing the model"
        ## define some variables for training
        n_train = X.shape[0]
        # number of times to go through the training data
        max_epochs = 1000
        # work on 20 training examples at a time
        batch_size = 20
        n_batches = int(np.ceil(float(n_train)/batch_size))
        
        mean_train_error = []
        ## do the actual training of the model
        for e in range(max_epochs):
            if not e or not (e+1) % 25:
                print("Epoch %i" % (e+1))
            train_error = []
            for bi in range(n_batches):
                # print("Batch %i / %i" % (bi+1, n_batches))
                if len(Y.shape) == 1:
                    mini_y = Y[bi*batch_size:min((bi+1)*batch_size,n_train)]
                else:
                    mini_y = Y[bi*batch_size:min((bi+1)*batch_size,n_train),:]
                mini_x = X[bi*batch_size:min((bi+1)*batch_size,n_train),:]
                # train model
                train_error.append(self.train_model(mini_x, mini_y))
            mean_train_error.append(np.mean(train_error))
            if not e or not (e+1) % 25:
                print("Mean training error: %f" % mean_train_error[-1])
        print("Final training error: %f" % mean_train_error[-1])

    def transform(self, X):
        """
        apply the model to some test data

        returns Y for all given X
        """
        assert X.shape[1] == self.x_dim, "number of features doesn't match model architecture"
        return self.predict(X)

    def score(self, X, Y):
        """
        given some input X and targets Y, compute the error (is returned as a float)
        """
        assert X.shape[0] == Y.shape[0], "need labels for all test examples"
        assert X.shape[1] == self.x_dim, "number of features doesn't match model architecture"
        return self.test_model(X, Y)
