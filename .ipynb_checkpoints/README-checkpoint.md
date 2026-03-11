# word2vec-from-scratch
## Author: Virginia Antón 

skip-gram model implementation from scratch using NumPy.  
This project implements a variant of Word2Vec model  including the full optimization procedure: forward pass, loss computation, gradient calculation, and parameter updates.

### Requirements
numpy
pandas
matplotlib
pickle

### Project structure
- 01_preprocesssing.ipynb for data loading, tokenisation, pair generation
- 02_skipgram.ipynb for model implementaion, training, evaluation
  
The notebooks are named in the order they should be run. 

### Dataset
Bedtime stories from Hugging Face, 199 short children's stories un English with simple vocabulary and clear sentence structures. With this dataset it is built a vocabulary of 1,497 unique words and 38,798 training pairs. 

### Implementation
First notebook loads the dataset, lowercases and tokenises the corpus, builds the mappings and generates the pairs. 

Second notebook implements the skip-gram model using a Python class with the following methods:
- __init__: Initialises W1, W2 and hyperparameters
- sigmoid(x): Defines a numerically stable sigmoid
- generateNegativeSamples(target): Uniform sampling of k noise words, excluding the target
- forwardprop(target, context,negSamples): Computes dot products and sigmoid scores
- loss(sigPos, sigNeg): Negative sampling binary cross-entropy loss
- backwardprop(...): Computes gradients and updates W1, W2
- step(target, context): Makes the updates

### Hyperparameter Search
Three hyperparameters are evaluated independently: Learning Rate, Embedding Dimension, Number of Negative Samples. 

### Evaluation
Applied cosine similarity to find nearest neighbors on W1 embeddings. 
Thanks to the evaluation, found some limitations and future work imporvements. 