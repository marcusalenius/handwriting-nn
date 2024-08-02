# Handwriting Recognizer

### A feedforward neural network implemented from scratch in Python. Reimplemented in PyTorch and improved using cross-entropy loss and regularization. Can recognize handwritten digits with 98% accuracy.

* `loader.py` contains the code to load the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and prepare it for the NumPy-based network and the PyTorch networks.
* `network.py` contains the code for a neural network implemented from scratch. It is based on the work in Michael Nielsen’s book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com).
* `torch_network.py` contains the code for three neural networks implemented in PyTorch.
* `train.py` trains the NumPy network and the three PyTorch networks.
* `recognize.py` contains code to recognize digits in custom images and write the results to a CSV file.

#### By [Marcus Alenius](https://www.linkedin.com/in/marcusalenius/)