# -----------------------------------------------------------------------------
# Code for a feedforward neural network implemented from scratch. Based on 
# Michael Nielsen's excellent book 'Neural Networks and Deep Learning' 
# http://neuralnetworksanddeeplearning.com. 
#
# Marcus Alenius, 2024
# -----------------------------------------------------------------------------

import random
import numpy as np

class Network: 

    def __init__(self, sizes):
        '''
        Creates an instance of `Network`. Takes `sizes`, which contains the number of 
        neurons in each layer. 
        '''
        self.num_layers = len(sizes)
        # `self.biases` contains n random numbers for each layer except the first
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]
        # `self.weights` contains y * x random numbers for each layer pair, where x is
        # the 'from' layer and y is the 'to' layer.
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]
    
    def __call__(self, x):
        '''
        Returns the output of the network (a 10-dimensional unit vector) 
        given an input `x`.
        '''
        return self.feedforward(x)
    
    def feedforward(self, x):
        '''
        Returns the output given an input `x` for the network.
        '''
        # apply a' = sigmoid(w a + b) for each layer
        a = x
        for (b, w) in zip(self.biases, self.weights):
            a = Network.sigmoid(np.dot(w, a) + b)
        return a

    def train(self, train_data, num_epochs, mini_batch_size, lr, 
              test_data=None):
        '''
        Trains the network using stochastic gradient descent and the quadratic cost
        function, with mini_batches of size `mini_batch_size` and `num_epochs` epochs. 
        `lr` is the learning rate. If `test_data` is provided, the accuracy will be 
        calculated after each epoch.
        '''
        for epoch in range(num_epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[i:i+mini_batch_size]
                            for i in range(0, len(train_data), mini_batch_size)]
            for mini_batch in mini_batches:
                # for each mini_batch, apply a single step of gradient descent
                self.update_weights_biases(mini_batch, lr)
            if test_data is not None: 
                print(f'Epoch {epoch}: Accuracy: {self.get_accuracy(test_data):.2f}%')
            else:
                print(f'Epoch {epoch} complete')
    
    def update_weights_biases(self, mini_batch, lr):
        '''
        Updates the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch. `mini_batch` is a list of 
        `(x, y)` tuples, and `lr` is the learning rate.
        '''
        # initialize lists to hold the bias and weight components of the gradient vector
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in mini_batch:
            # update nabla_b, nabla_w with the results of the backprop
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for (nw, dnw) in zip(nabla_w, delta_nabla_w)]
        # update weights and biases (formula 20 and 21 
        #    http://neuralnetworksanddeeplearning.com/chap1.html)
        self.weights = [(w - (lr / len(mini_batch)) * nw)
                        for (w, nw) in zip(self.weights, nabla_w)]
        self.biases = [(b - (lr / len(mini_batch)) * nb)
                        for (b, nb) in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        Returns the tuple `(nabla_b, nabla_w)` representing the gradient for the 
        cost function. `nabla_b` and `nabla_w` are layer-by-layer lists of numpy 
        arrays, similar to `self.biases` and `self.weights`.
        '''
        # initialize lists to hold the bias and weight components of the gradient vector
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []            # list to store all the z vectors, layer by layer
        for (b, w) in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Network.sigmoid(z)
            activations.append(activation)
        # backward pass
        # compute the output error (formula BP1
        #    http://neuralnetworksanddeeplearning.com/chap2.html)
        delta = (Network.cost_derivative(activations[-1], y) * 
                 Network.sigmoid_prime(zs[-1]))
        # compute the last layer of nabla_b and nabla_w (formula BP3 and BP4
        #    http://neuralnetworksanddeeplearning.com/chap2.html)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # backpropagate the error 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Network.sigmoid_prime(z)
            # compute the error according to BP2 
            #   (http://neuralnetworksanddeeplearning.com/chap2.html)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # compute nabla_b and nabla_w for this layer (formula BP3 and BP4
            #    http://neuralnetworksanddeeplearning.com/chap2.html)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def get_accuracy(self, test_data):
        '''
        Calculates the accuracy on test data. That is the number of correct
        predictions divided by the total number of data.
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        num_correct = sum(int(result == y) for (result, y) in test_results)
        return (num_correct / (len(test_data))) * 100
    
    @staticmethod
    def sigmoid(z):
        '''
        The sigmoid activation function. When `z` is a vector, Numpy automatically
        applies the sigmoid function elementwise, that is, in vectorized form.
        '''
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        '''
        The derivative of the sigmoid function.
        '''
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))
    
    @staticmethod
    def cost_derivative(output_activations, y):
        '''
        The derivative of the quadratic cost function.
        '''
        return 2 * (output_activations - y)
