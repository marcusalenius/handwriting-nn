# -----------------------------------------------------------------------------
# Code for three feedforward neural networks implemented in PyTorch.
# Referenced https://www.learnpytorch.io and https://pytorch.org/docs/stable.
# 
# Marcus Alenius, 2024
# -----------------------------------------------------------------------------

import torch
from torch import nn 
from torch.utils.data import DataLoader

class TorchNetwork: 

    def __init__(self, sizes):
        '''
        Creates an instance of `Network`. Takes `sizes`, which contains the number of 
        neurons in each layer. 
        '''
        layers = [nn.Flatten()]
        for (in_features, out_features) in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    
    def __call__(self, x):
        '''
        Returns the output of the network (a 10-dimensional unit vector) 
        given an input `x`.
        '''
        self.model.eval()
        with torch.inference_mode():
            output = self.model(x)
        return output

    def torch_train(self, loss_fn, optimizer, 
                    train_data, num_epochs, mini_batch_size, lr, 
                    test_data=None):
        '''
        Generic train method to be called by the subclass's `train` method.
        '''
        # split into mini-batches, creates an iterable 
        train_dataloader = DataLoader(train_data, batch_size=mini_batch_size, 
                                      shuffle=True)
        # training loop
        for epoch in range(num_epochs):
            # iterates through the mini-batches 
            for (x, y) in train_dataloader:
                self.model.train()
                y_pred = self.model(x)
                # we need to vectorize `y` to get in the same format as `y_pred`
                loss = loss_fn(y_pred, Network0.vectorize_target_batch(y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if test_data is not None: 
                print(f'Epoch {epoch}: Accuracy: {self.get_accuracy(test_data):.2f}%')
            else:
                print(f'Epoch {epoch} complete')

    @staticmethod
    def vectorize_target_batch(y_batch):
        '''
        Returns a tensor with a 10-dimensional unit vector with a `1.0` in 
        the `y`th position and zeroes elsewhere, for each `y` in `y_batch`.
        '''
        result_vectors = []
        for y in y_batch: 
            vector = [0 for _ in range(10)]
            vector[y.item()] = 1.0
            result_vectors.append(vector)
        return torch.tensor(result_vectors, dtype=torch.float32)

    def get_accuracy(self, test_data):
        '''
        Calculates the accuracy on test data. That is the number of correct predictions 
        divided by the total number of data.
        '''
        test_results = [(torch.argmax(self.model(x)), y)
                        for (x, y) in test_data]
        num_correct = sum(int(result == y) for (result, y) in test_results)
        return (num_correct / (len(test_data))) * 100


class Network0(TorchNetwork):

    def __init__(self, sizes):
        '''
        Creates an instance of `Network0`, that is an instance that uses the 
        quadratic cost function and SGD. Takes `sizes`, which contains the 
        number of neurons in each layer. 
        '''
        super().__init__(sizes)
    
    def train(self, train_data, num_epochs, mini_batch_size, lr, 
              test_data=None):
        '''
        Trains the network using stochastic gradient descent and the quadratic cost
        function, with mini_batches of size `mini_batch_size` and `num_epochs` epochs. 
        `lr` is the learning rate. If `test_data` is provided, the accuracy will be 
        calculated after each epoch.
        '''
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr)
        self.torch_train(loss_fn, optimizer, 
                         train_data, num_epochs, mini_batch_size, lr, 
                         test_data=test_data)


class Network1(TorchNetwork):

    def __init__(self, sizes):
        '''
        Creates an instance of `Network1`, that is an instance that uses 
        cross entropy loss and SGD. Takes `sizes`, which contains the 
        number of neurons in each layer. 
        '''
        super().__init__(sizes)
    
    def train(self, train_data, num_epochs, mini_batch_size, lr, 
              test_data=None):
        '''
        Trains the network using stochastic gradient descent and cross entropy
        loss, with mini_batches of size `mini_batch_size` and `num_epochs` epochs. 
        `lr` is the learning rate. If `test_data` is provided, the accuracy will 
        be calculated after each epoch.
        '''
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr)
        self.torch_train(loss_fn, optimizer, 
                         train_data, num_epochs, mini_batch_size, lr, 
                         test_data=test_data)


class Network2(TorchNetwork):

    def __init__(self, sizes):
        '''
        Creates an instance of `Network2`, that is an instance that uses 
        cross entropy loss and SGD with L2 regularization. Takes `sizes`,
        which contains the number of neurons in each layer. 
        '''
        super().__init__(sizes)

    def train(self, train_data, num_epochs, mini_batch_size, lr, weight_decay,
              test_data=None):
        '''
        Trains the network using stochastic gradient descent and cross entropy
        loss, with mini_batches of size `mini_batch_size` and `num_epochs` epochs. 
        `lr` is the learning rate and `weight_decay` is the weight decay. If 
        `test_data` is provided, the accuracy will be calculated after each epoch.
        '''
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr, 
                                    weight_decay=weight_decay)
        self.torch_train(loss_fn, optimizer, 
                         train_data, num_epochs, mini_batch_size, lr, 
                         test_data=test_data)
