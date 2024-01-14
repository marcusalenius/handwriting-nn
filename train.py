# -----------------------------------------------------------------------------
# Code to train the Numpy network and the three PyTorch networks.
#
# Marcus Alenius, 2024
# -----------------------------------------------------------------------------

import loader
import network
import torch_network

# load data for Numpy network
train_data, test_data = loader.get_data()

# load data for PyTorch networks
train_data_torch, test_data_torch = loader.get_data_for_torch()

# Numpy network
net = network.Network([784, 30, 10])
print('Training Numpy network:')
net.train(train_data, 30, 10, 3.0, test_data=test_data)

# PyTorch network 0 -- quadratic cost and SGD
net0 = torch_network.Network0([784, 30, 10])
print('Training PyTorch network 0:')
net0.train(train_data_torch, 30, 10, 3.0, test_data=test_data_torch)

# PyTorch network 1 -- quadratic cost and SGD
net1 = torch_network.Network1([784, 140, 10])
print('Training PyTorch network 1:')
net1.train(train_data_torch, 30, 10, 3.0, test_data=test_data_torch)

# PyTorch network 2 -- quadratic cost and SGD with L2 regularization
net2 = torch_network.Network2([784, 140, 10])
print('Training PyTorch network 2:')
net2.train(train_data_torch, 30, 10, 3.0, 1e-6, test_data=test_data_torch)
