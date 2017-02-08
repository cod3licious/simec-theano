This repository contains the code for the Similarity Encoder (SimEc) neural network model (and others). To get an overview, first check out the iPython notebook at `examples_simec.ipynb`. 
For further information see: http://arxiv.org/abs/1702.01824

dependencies: (main code) numpy, theano; (experiments) scipy, sklearn, matplotlib, [nlputils](https://github.com/cod3licious/nlputils)

### main code
- `ann.py` contains basic theano code to generate a neural network with multiple layers (see also here: http://deeplearning.net/tutorial/index.html).
- `ann_models.py` builds upon `ann.py` to create a neural network model, which can be used to solve supervised ML tasks like classification or regression with a similar interface as sklearn classifiers.
- `similarity_encoder.py` contains the SimilarityEncoder class to set up, train, and apply the similarity encoder; by default it sets up a linear SimEc, i.e. with one linear layer to project the high dimensional data to a low dimensional embedding, and another layer to compute the output, the approximated similarities. Depending on the target similarities that should be preserved in the embedding space, more complex (i.e. deeper) models and possibly the optimization of a different cost function than the mean squared error can be beneficial (see also the notebook with examples).

### experiments
- `ann_test.py` contains some exemplary classification and regression problems to test the `ann_models` neural networks.
- `utils.py` contains some helper functions mostly to load datasets and plot results used by `examples_simec.ipynb`.
- `examples_simec.ipynb` is an iPython notebook with multiple examples showing embeddings of image and text datasets created using standard dimensionality reduction algorithms (kPCA, Isomap, ...) as well as Similarity Encoder models with different network architectures, cost functions, etc.
- `qa_utils.py` contains functions to examine the quality of the embedding (how well the target similarities are preserved), e.g. by creating a Shepard's plot.


If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
