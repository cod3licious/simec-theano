This repository contains the code for the similarity encoder (SimEc) neural network model (and others). For further information see: http://openreview.net/forum?id=SkBsEQYll

dependencies: (main code) numpy, theano; (experiments) scipy, sklearn, matplotlib, [nlputils](https://github.com/cod3licious/nlputils)

### main code
- `ann.py` contains basic theano code to generate a neural network with multiple layers and various activation functions (see also here: http://deeplearning.net/tutorial/index.html)
- `ann_models.py` builds upon `ann.py` to create a neural network model which can be used to solve supervised ML tasks like classification or regression with a similar interface as sklearn classifiers.
- `similarity_encoder.py` contains the SimilarityEncoder class to set up, train, and apply the similarity encoder; right now it is natively set up to have a single weight matrix to project the input to the embedding layer and another to project the embedding to the output layer to estimate the error, more layers and different activation functions would make sense for more complicated kernels than the linear one though

### experiments
- `ann_test.py` contains some exemplary classification and regression problems to test the `ann_models` neural networks
- `utils.py` contains some helper functions mostly to load datasets and plot results used by `examples_simec.ipynb`
- `examples_simec.ipynb` is an iPython notebook with multiple examples showing embeddings created using various standard embedding methods (kPCA, Isomap, ...) and the Similarity Encoder.


If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
