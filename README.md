Similarity Encoders (SimEc)
===========================

**This library is deprecated! Check out the [`keras`-based SimEc code](https://github.com/cod3licious/simec) instead**

This repository contains the code for the Similarity Encoder (SimEc) neural network model (and others) based on the `theano` library. Have a look at the [Jupyter notebook with examples](https://github.com/cod3licious/simec-theano/blob/master/examples_simec.ipynb)! 
For further details on the model and experiments please refer to the [paper](http://arxiv.org/abs/1702.01824).

This code is still work in progress and intended for research purposes. It was programmed for Python 2.7, but should theoretically also run on newer Python 3 versions - no guarantees on this though (open an issue if you find a bug, please)!


simec library components
------------------------

dependencies: `numpy`, `theano` (and `future` on Python 2.7)

- `ann.py` contains basic theano code to generate a neural network with multiple layers (see also the [theano tutorial](http://deeplearning.net/tutorial/index.html)).
- `ann_models.py` builds upon `ann.py` to create a neural network model, which can be used to solve supervised ML tasks like classification or regression with a similar interface as sklearn classifiers.
- `similarity_encoder.py` contains the SimilarityEncoder class to set up, train, and apply the similarity encoder; by default it sets up a linear SimEc, i.e. with one linear layer to project the high dimensional data to a low dimensional embedding, and another layer to compute the output, the approximated similarities. Depending on the target similarities that should be preserved in the embedding space, more complex (i.e. deeper) models should be used (see also the notebook with examples).

examples
--------

additional dependencies: `scipy`, `sklearn`, `matplotlib`, [`nlputils`](https://github.com/cod3licious/nlputils) 

- `ann_test.py` contains some exemplary classification and regression problems to test the `ann_models` neural networks.
- `utils.py` contains some helper functions mostly to load datasets and plot results, used by `examples_simec.ipynb`.
- `examples_simec.ipynb` is an iPython notebook with multiple examples showing embeddings of image and text datasets created using standard dimensionality reduction algorithms (kPCA, Isomap, ...) as well as Similarity Encoder models with different network architectures, etc.

If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
