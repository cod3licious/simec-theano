import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import theano
import theano.tensor as T
from sklearn.datasets import make_moons, make_circles, make_classification

from ann import *

def linear_regression(y_dim=5, x_dim=10):
    ## generate training and test data
    n_train = 1000
    W = np.random.randint(-4,5,size=(x_dim,y_dim))
    b = np.random.randint(-2,2,size=(1,y_dim))
    X = np.random.rand(n_train,x_dim)
    Y = np.dot(X,W)+b
    X += np.random.randn(n_train,x_dim)*0.02
    X_test = np.random.rand(20,x_dim)
    Y_test = np.dot(X_test,W)+b
    X_test += np.random.randn(20,x_dim)*0.02 

    ## build the model
    # some parameters
    learning_rate = 0.001
    L1_reg = 0.00
    L2_reg = 0.001

    # allocate symbolic variables for the data
    x = T.matrix('x')    # input data
    y = T.matrix('y')    # corresponding labels

    # construct the ANN
    classifier = ANN(
        x_in=x,
        n_in=x_dim,
        n_out=[y_dim],
        activation=[],
        seed=12
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2)
    cost = (
        mean_squared_error(classifier.output, y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compile a Theano function that computes the error on some test data
    test_model = theano.function(
        inputs=[x, y],
        outputs=mean_squared_error(classifier.output, y)
    )

    predict = theano.function(
        inputs=[x],
        outputs=classifier.output
    )

    # compute the gradient of cost with respect to all parameters
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compile a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[x, y],
        outputs=mean_squared_error(classifier.output, y),
        updates=updates
    )

    ## define some variables for training
    # number of times to go through the training data
    max_epochs = 1000
    # work on 20 training examples at a time - they are really large
    batch_size = 20
    # load all the labels
    n_batches = int(np.ceil(float(n_train)/batch_size))
    
    # initial error
    k = test_model(X_test, Y_test)
    print("### Initial Test Error: %f" % k)
    ## do the actual training of the model
    for e in range(max_epochs):
        print("Epoch %i" % (e+1))
        train_error = []
        for bi in range(n_batches):
            # print("Batch %i / %i" % (bi+1, n_batches))
            mini_y = Y[bi*batch_size:min((bi+1)*batch_size,n_train),:]
            mini_x = X[bi*batch_size:min((bi+1)*batch_size,n_train),:]
            # train model
            train_error.append(train_model(mini_x, mini_y))
            #print("Cost: %.3f" % train_error[-1])
        print("Mean training error: %f" % np.mean(train_error))
        # validate
        k = test_model(X_test, Y_test)
        print("### Test Error: %f" % k)

    if x_dim == 1 and y_dim == 1:
        plt.figure()
        plt.plot(X[:,0],Y[:,0],'m*')
        X_plot = np.linspace(np.min(X),np.max(X),1000)
        plt.plot(X_plot,predict(X_plot[:,np.newaxis]),'k')
        plt.plot(X_plot,np.dot(X_plot[:,np.newaxis],W)+b,'b')
        plt.xlabel('x')
        plt.ylabel('y')


def nonlinear_regression(y_dim=1, x_dim=1):
    ## generate training and test data
    n_train = 1000
    X = np.random.rand(n_train,x_dim)*np.pi*2.
    Y = np.sin(X)
    X += np.random.randn(n_train,x_dim)*0.02
    X_test = np.random.rand(20,x_dim)*np.pi*2.
    Y_test = np.sin(X_test)
    X_test += np.random.randn(20,x_dim)*0.02 

    ## build the model
    # some parameters
    learning_rate = 0.001
    L1_reg = 0.00
    L2_reg = 0.001

    # allocate symbolic variables for the data
    x = T.matrix('x')    # input data
    y = T.matrix('y')    # corresponding labels

    # construct the ANN
    classifier = ANN(
        x_in=x,
        n_in=x_dim,
        n_out=[100, 50, y_dim],
        activation=[T.tanh, T.tanh, None],
        seed=12
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2)
    cost = (
        mean_squared_error(classifier.output, y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compile a Theano function that computes the error on some test data
    test_model = theano.function(
        inputs=[x, y],
        outputs=mean_squared_error(classifier.output, y)
    )

    predict = theano.function(
        inputs=[x],
        outputs=classifier.output
    )

    # compute the gradient of cost with respect to all parameters
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compile a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[x, y],
        outputs=mean_squared_error(classifier.output, y),
        updates=updates
    )

    ## define some variables for training
    # number of times to go through the training data
    max_epochs = 1000
    # work on 20 training examples at a time - they are really large
    batch_size = 20
    # load all the labels
    n_batches = int(np.ceil(float(n_train)/batch_size))
    
    # initial error
    k = test_model(X_test, Y_test)
    print("### Initial Test Error: %f" % k)
    ## do the actual training of the model
    for e in range(max_epochs):
        print("Epoch %i" % (e+1))
        train_error = []
        for bi in range(n_batches):
            # print("Batch %i / %i" % (bi+1, n_batches))
            mini_y = Y[bi*batch_size:min((bi+1)*batch_size,n_train),:]
            mini_x = X[bi*batch_size:min((bi+1)*batch_size,n_train),:]
            # train model
            train_error.append(train_model(mini_x, mini_y))
            #print("Cost: %.3f" % train_error[-1])
        print("Mean training error: %f" % np.mean(train_error))
        # validate
        k = test_model(X_test, Y_test)
        print("### Test Error: %f" % k)

    if x_dim == 1 and y_dim == 1:
        plt.figure()
        plt.plot(X[:,0],Y[:,0],'m*')
        X_plot = np.linspace(np.min(X),np.max(X),1000)
        plt.plot(X_plot,predict(X_plot[:,np.newaxis]),'k')
        plt.plot(X_plot,np.sin(X_plot),'b')
        plt.xlabel('x')
        plt.ylabel('y')


def classification(dataset=0):
    ## generate training and test data
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
        print "dataset unknown"
        return

    ## build the model
    # some parameters
    learning_rate = 0.01
    L1_reg = 0.00
    L2_reg = 0.00

    # allocate symbolic variables for the data
    x = T.matrix('x')     # input data
    y = T.ivector('y')    # corresponding labels

    # construct the ANN
    classifier = ANN(
        x_in=x,
        n_in=X.shape[1],
        n_out=[100, 50, 2],
        activation=[T.tanh, T.tanh, T.nnet.softmax],
        seed=12
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2)
    cost = (
        negative_log_likelihood(classifier.output, y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compile a Theano function that computes the error on some test data
    test_model = theano.function(
        inputs=[x, y],
        outputs=zero_one_loss(T.argmax(classifier.output, axis=1), y),
        allow_input_downcast=True
    )

    predict = theano.function(
        inputs=[x],
        outputs=classifier.output
    )

    # compute the gradient of cost with respect to all parameters
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compile a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[x, y],
        outputs=zero_one_loss(T.argmax(classifier.output, axis=1), y),
        updates=updates,
        allow_input_downcast=True
    )

    ## define some variables for training
    # number of times to go through the training data
    max_epochs = 1000
    # work on 20 training examples at a time - they are really large
    batch_size = 20
    # load all the labels
    n_batches = int(np.ceil(float(n_train)/batch_size))
    
    # initial error
    k = test_model(X_test, Y_test)
    print("### Initial Test Error: %f" % k)
    ## do the actual training of the model
    for e in range(max_epochs):
        print("Epoch %i" % (e+1))
        train_error = []
        for bi in range(n_batches):
            # print("Batch %i / %i" % (bi+1, n_batches))
            mini_y = Y[bi*batch_size:min((bi+1)*batch_size,n_train)]
            mini_x = X[bi*batch_size:min((bi+1)*batch_size,n_train),:]
            # train model
            train_error.append(train_model(mini_x, mini_y))
            #print("Cost: %.3f" % train_error[-1])
        print("Mean training error: %f" % np.mean(train_error))
        # validate
        k = test_model(X_test, Y_test)
        print("### Test Error: %f" % k)

    ### plot dataset + predictions
    plt.figure()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    Z = predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm_bright,
               alpha=0.6)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())


if __name__ == '__main__':
    linear_regression(1, 1)
    nonlinear_regression(1, 1)
    classification(0)
    classification(1)
    classification(2)
    plt.show()
