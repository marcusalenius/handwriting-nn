# -----------------------------------------------------------------------------
# Code to load the MNIST dataset using PyTorch and prepare it for the 
# Numpy-based network and the PyTorch network.
#
# Marcus Alenius, 2024
# -----------------------------------------------------------------------------

from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

def get_data():
    '''
    Returns the tuple `(train_data, test_data)`, to be used in the 
    Numpy-based network.
    
    `train_data` is a list containing 60,000 `(image, target_vector)`
    tuples. `image` is a 784-dimensional `numpy.ndarray` containing the
    input image. `target_vector` is a 10-dimensional `numpy.ndarray`
    representing the unit vector corresponding to the correct digit.

    `test_data` is a list containing 10,000 `(image, target)` tuples. 
    `image` is the same as above, but `target` is just the correct digit.
    '''
    # load the MNIST dataset using PyTorch
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(), 
        target_transform=None 
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(), 
        target_transform=None 
    )
    # extract the images from the training data
    train_data_image_array = train_data.data.numpy().astype('float64')
    train_data_image_reshaped = [np.reshape(x, (784, 1)) 
                                 for x in train_data_image_array]
    # pixel values are between [0, 255], we want them between [0, 1]
    train_data_image_scaled = [image / 255 for image in train_data_image_reshaped]

    # extract the targets from the training data
    train_data_target_array = train_data.targets.numpy().astype('float64')

    # combine images and targets
    train_data_all = list(zip(train_data_image_scaled, 
                              train_data_target_array))
    
    # vectorize the targets
    train_data_final = [(image, vectorize_target(target)) 
                        for (image, target) in train_data_all]
    

    # similarly for the test data
    test_data_image_array = test_data.data.numpy().astype('float64')
    test_data_image_reshaped = [np.reshape(x, (784, 1)) 
                                for x in test_data_image_array]
    test_data_image_scaled = [image / 255 for image in test_data_image_reshaped]
    test_data_target_array = test_data.targets.numpy().astype('float64')
    test_data_final = list(zip(test_data_image_scaled, 
                               test_data_target_array))

    return train_data_final, test_data_final

def vectorize_target(y):
    '''
    Returns a 10-dimensional unit vector with a `1.0` in the `y`th
    position and zeroes elsewhere.
    '''
    vector = np.zeros((10, 1))
    vector[int(y)] = 1.0
    return vector

def get_data_for_torch():
    '''
    Returns the tuple `(train_data, test_data)`, to be used in the 
    PyTorch network. `train_data` and `test_data` are both 
    torchvision dataset objects.
    '''
    # load the MNIST dataset using PyTorch
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(), 
        target_transform=None 
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(), 
        target_transform=None 
    )

    return train_data, test_data