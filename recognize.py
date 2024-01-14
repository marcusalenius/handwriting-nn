# -----------------------------------------------------------------------------
# Contains code to recognize the digits in 28x28 pixel images, located in the
# ./images subdirectory.
# 
# Marcus Alenius, 2024
# -----------------------------------------------------------------------------

from PIL import Image
import os, os.path
import csv
import numpy as np
import torch
from torchvision import transforms


def recognize_digits(network, torch=False):
    '''
    Takes `network`, a trained, callable neural network that produces a 
    10-dimensional unit vector when called on a 784-dimensional vector 
    input. The flag `torch` indicates whether it is a PyTorch network.
    Writes a csv file with rows `file_name, predicted_digit`.
    '''
    # dict to map `file_name` to `predicted_digit`
    output = dict()

    # predict digits for each image
    path = './images'
    valid_file_formats = ['.jpg', '.png']
    files = os.listdir(path)
    for file_name in files:
        _, extension = os.path.splitext(file_name)
        if extension not in valid_file_formats:
            continue
        predicted_digit = predict_digit(os.path.join(path, file_name), network, torch=torch)
        output[file_name] = predicted_digit
    
    # write output dict to csv
    with open('recognized_digits.csv', 'w') as f:
        writer = csv.writer(f)
        for file_name, predicted_digit in sorted(output.items()):
            writer.writerow([file_name, predicted_digit])


def predict_digit(image_file_path, network, torch):
    '''
    Takes `image_file_path`, a path to an image, `network`, a trained neural 
    network, and `torch`, a flag indicating whether it is a PyTorch network. 
    Returns the digit the network predicts based on the input image.
    '''
    # open image and convert it to single channel image (grayscale)
    #   'L' means that the luminance is stored
    image = Image.open(image_file_path).convert('L')

    if torch: 
        image_tensor = transforms.PILToTensor()(image)
        # pixel values are between [0, 255], we want them between [0, 1]
        image_tensor = image_tensor / 255
        predicted_digit = devectorize_output(network(image_tensor))
    else:
        image_arr = np.array(image)
        # pixel values are between [0, 255], we want them between [0, 1]
        image_arr = image_arr / 255
        # image_arr is size (28, 28), we want (784, 1)
        image_arr_reshaped = np.reshape(image_arr, (784, 1))
        predicted_digit = devectorize_output(network(image_arr_reshaped))

    return predicted_digit


def devectorize_output(output):
    '''
    Takes a 10-dimensional output unit vector `output`
    and returns the corresponding digit result.
    '''
    if torch.is_tensor(output):
        return torch.argmax(output).item()
    else:
        return np.argmax(output)



####################

import torch_network
import loader

train_data_torch, test_data_torch = loader.get_data_for_torch()
net2 = torch_network.Network2([784, 140, 10])
print('Training PyTorch network 2:')
net2.train(train_data_torch, 30, 10, 3.0, 1e-6, test_data=test_data_torch)

recognize_digits(net2, torch=True)