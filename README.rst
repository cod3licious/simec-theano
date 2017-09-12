Similarity Encoders (SimEc)
===========================

This repository contains the code for the Similarity Encoder (SimEc) neural network model (and others). To get an overview, first check out the iPython notebook at |examples/examples_simec.ipynb|_. 
For further details on the model and experiments please refer to the paper_ (and of course please consider citing it ;-)).

.. |examples/examples_simec.ipynb| replace:: ``examples/examples_simec.ipynb``
.. _examples/examples_simec.ipynb: https://github.com/cod3licious/simec/blob/master/examples/examples_simec.ipynb
.. _paper: http://arxiv.org/abs/1702.01824


This code is still work in progress and intended for research purposes. It was programmed for Python 2.7, but should theoretically also run on newer Python 3 versions - no guarantees on this though (open an issue if you find a bug, please)!


installation
------------
You either download the code from here and include the simec folder in your ``$PYTHONPATH`` or install (the library components only) via pip:

    ``$ pip install simec``


simec library components
------------------------

dependencies: numpy, theano

- ``ann.py`` contains basic theano code to generate a neural network with multiple layers (see also the `theano tutorial`_).
- ``ann_models.py`` builds upon ``ann.py`` to create a neural network model, which can be used to solve supervised ML tasks like classification or regression with a similar interface as sklearn classifiers.
- ``similarity_encoder.py`` contains the SimilarityEncoder class to set up, train, and apply the similarity encoder; by default it sets up a linear SimEc, i.e. with one linear layer to project the high dimensional data to a low dimensional embedding, and another layer to compute the output, the approximated similarities. Depending on the target similarities that should be preserved in the embedding space, more complex (i.e. deeper) models should be used (see also the notebook with examples).

.. _`theano tutorial`: http://deeplearning.net/tutorial/index.html


examples
--------

additional dependencies: scipy, sklearn, matplotlib, nlputils_

.. _nlputils: https://github.com/cod3licious/nlputils

- ``ann_test.py`` contains some exemplary classification and regression problems to test the ``ann_models`` neural networks.
- ``utils.py`` contains some helper functions mostly to load datasets and plot results, e.g. used by ``examples_simec.ipynb``.
- ``examples_simec.ipynb`` is an iPython notebook with multiple examples showing embeddings of image and text datasets created using standard dimensionality reduction algorithms (kPCA, Isomap, ...) as well as Similarity Encoder models with different network architectures, cost functions, etc.
- ``classify_cancer.py`` contains code to classify the `cancer papers dataset`_ with different SimEc features.

.. _`cancer papers dataset`: https://github.com/cod3licious/cancer_papers

If you have any questions please don't hesitate to send me an `email <mailto:cod3licious@gmail.com>`_ and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
