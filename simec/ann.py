from __future__ import division
from builtins import object
import numpy as np
import theano
import theano.tensor as T
from theano import sparse


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
        if activation is None:
            W_values = rng.randn(
                n_in, n_out
            ) * 0.01
        else:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

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
        # apply the activation function (if any)
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
            assert len(n_out) == len(activation), "need as many activation functions as layers"
        rng = np.random.RandomState(seed)
        # create all the layers
        self.layers = []
        self.params = []
        for i, n in enumerate(n_out):
            # layers get as input x_in or the output of the previous layer
            self.layers.append(
                NNLayer(x_in if not self.layers else self.layers[-1].output,
                        n_out[i - 1] if i else n_in, n,
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
        self.orthNN = sum([T.abs_(T.dot(l.W.T, l.W) - T.nlinalg.diag(T.nlinalg.diag(T.dot(l.W.T, l.W)))).sum() /
                           float(l.W.get_value().shape[1]) for l in self.layers[:1]]) / float(len(self.layers[:1]))
        # orthogonalization of weights from embedding to output as YY^T is the eigendecomposition, i.e. W_1 should be orthogonal
        # normalize by 1/(d**2-d) to be independent of the dimensionality of the embedding
        d = self.layers[-1].W.get_value().shape[0]
        self.orthOT = T.abs_(T.dot(self.layers[-1].W, self.layers[-1].W.T) - T.nlinalg.diag(
            T.nlinalg.diag(T.dot(self.layers[-1].W, self.layers[-1].W.T)))).sum() / float(d * d - d)
        # unit length weights in the last layer
        self.normOT = T.abs_(1. - T.sqrt((self.layers[-1].W ** 2).sum(axis=0))).sum() / float(self.layers[-1].W.get_value().shape[1])
