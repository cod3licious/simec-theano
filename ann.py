import numpy as np
import theano
import theano.tensor as T
from theano import sparse


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
        - y_pred: N dimensional vector containing the predicted classes
        - y_true: N dimensional vector containing the true class labels
    Returns:
        - zero-one-loss
    """
    # the T.neq operator returns a vector of 0s and 1s, where 1
    # represents a mistake in prediction
    return T.mean(T.neq(y_pred, y_true))


class NNLayer(object):
    """
    A single layer of a Neural Network architecture.
    Depending on the parameters, this can function as a hidden or output layer (e.g. LogReg).
    """
    def __init__(self, x_in, n_in, n_out, activation=None, rng=None, seed=0):
        """
        Initialize the layer.

        Inputs:
            - x_in: a symbolic theano variable describing the input
            - n_in: dimensions the input will have
            - n_out: dimensions the output should have
            - activation: non-linear activation function applied to the output (if any)
            - seed: used to initialize the random number generator
        """
        if rng is None:
            rng = np.random.RandomState(seed)
        # initialize the weights - optimal values depend on the activation function
        if activation in [T.tanh, T.nnet.sigmoid]:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
        else:
            W_values = rng.randn(
                n_in, n_out
            )*0.01
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        # compute the output
        if isinstance(x_in.type, sparse.type.SparseType):
            lin_output = sparse.dot(x_in, self.W) + self.b
        else:
            lin_output = T.dot(x_in, self.W) + self.b
        self.output = (
            lin_output if not activation
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class ANN(object):
    """
    Artificial Neural Network with one or multiple layers of different types
    """
    def __init__(self, x_in, n_in, n_out, activation=[], seed=0):
        """
        Initialize the neural network

        Inputs:
            - x_in: symbolic variable representing the input to the network
            - n_in: number of dimensions the input will have
            - n_out: a list with the number of units the hidden/output layers should have
            - activation: if any, the activation functions applied to the hidden/output layers
            - seed: initial random seed used for the initialization of the layers
        """
        if activation:
            assert len(n_out)==len(activation), "need as many activation functions as layers"
        rng = np.random.RandomState(seed)
        # create all the layers
        self.layers = []
        self.params = []
        for i, n in enumerate(n_out):
            # layers get as input x_in or the output of the previous layer
            self.layers.append(
                NNLayer(x_in if not self.layers else self.layers[-1].output, 
                        n_out[i-1] if i else n_in, n, 
                        activation[i] if activation else None, rng)
                )
            self.params.extend(self.layers[-1].params)
        self.output = self.layers[-1].output
        # Define regularization
        # L1 norm
        self.L1 = sum([abs(l.W).sum() for l in self.layers])
        # square of L2 norm
        self.L2_sqr = sum([(l.W ** 2).sum() for l in self.layers])
        # orthogonalization of weights in the NN (probably not - only in the linear case)
        self.orthNN = sum([T.abs_(T.dot(l.W.T, l.W) - T.nlinalg.diag(T.nlinalg.diag(T.dot(l.W.T, l.W)))).sum()/float(l.W.get_value().shape[1]) for l in self.layers[:1]])/float(len(self.layers[:1]))
        # orthogonalization of weights from embedding to output as YY^T is the eigendecomposition, i.e. W_1 should be orthogonal
        self.orthOT = T.abs_(T.dot(self.layers[-1].W, self.layers[-1].W.T) - T.nlinalg.diag(T.nlinalg.diag(T.dot(self.layers[-1].W, self.layers[-1].W.T)))).sum()/float(self.layers[-1].W.get_value().shape[0])
