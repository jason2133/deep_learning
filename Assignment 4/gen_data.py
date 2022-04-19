# Generate simulation models, real datasets, ...
# Version 0.1
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import random
import math

def gen_twolayer1(N=5,D=4,H=10,C=3,dtype=torch.float32,device='cpu'):
    '''
    Inputs:
        N: # of observations
        D: dimension of the input
        H: # of hidden units
        C: # of classes
    Outputs:
        X: `dtype` tensor of shape (N, D) giving data points
        y: int64 tensor of shape (N,) giving labels, where each element is an
    integer in the range [0, C)
        params: A dictionary of toy model parameters, with keys:
        - 'W1': `dtype` tensor of shape (D, H) giving first-layer weights
        - 'b1': `dtype` tensor of shape (H,) giving first-layer biases
        - 'W2': `dtype` tensor of shape (H, C) giving second-layer weights
        - 'b2': `dtype` tensor of shape (C,) giving second-layer biases
    '''

    # Generate some random parameters, storing them in a dict
    params = {}
    params['W1'] = 1e-4 * torch.randn(D, H, device=device, dtype=dtype)
    params['b1'] = torch.zeros(H, device=device, dtype=dtype)
    params['W2'] = 1e-4 * torch.randn(H, C, device=device, dtype=dtype)
    params['b2'] = torch.zeros(C, device=device, dtype=dtype)

    # Generate some random inputs and labels
    X = 10.0 * torch.randn(N, D, device=device, dtype=dtype)
    y = torch.tensor([0, 1, 2, 2, 1], device=device, dtype=torch.int64)

    return X, y, params


def load_cifar10(dtype=torch.float32, cuda=True, preprocess=True, bias_trick=False):
    """
    Return the CIFAR10 dataset, automatically downloading it if necessary.

    Returns:
    - x_train: `x_dtype` tensor of shape (num_train, 3, 32, 32)
    - y_train: int64 tensor of shape (num_train, 3, 32, 32)
    - x_test: `x_dtype` tensor of shape (num_test, 3, 32, 32)
    - y_test: int64 tensor of shape (num_test, 3, 32, 32)
    """
    
    from torchvision.datasets import CIFAR10
    direc = 'cifar10/cifar-10-batches-py'
    download = not os.path.isdir(direc)
    trainset = CIFAR10(root=direc, train=True, download=download)
    testset =  CIFAR10(root=direc, train=False, download=download)

    X_train = torch.tensor(trainset.data, dtype=torch.float64).permute(0, 3, 1, 2).div_(255)
    y_train = torch.tensor(trainset.targets, dtype=torch.int64)

    X_test = torch.tensor(testset.data, dtype=torch.float64).permute(0, 3, 1, 2).div_(255)
    y_test = torch.tensor(testset.targets, dtype=torch.int64)
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # GPU
    if cuda:
      X_train = X_train.cuda()
      y_train = y_train.cuda()
      X_test = X_test.cuda()
      y_test = y_test.cuda()
    # preprocess
    if preprocess:
      # 1. Normalize the data: subtract the mean RGB (zero mean)
      mean_image = X_train.mean(dim=(0, 2, 3), keepdim=True)
      X_train -= mean_image
      X_test -= mean_image

      # 2. Reshape the image data into rows
      X_train = X_train.reshape(X_train.shape[0], -1)
      X_test = X_test.reshape(X_test.shape[0], -1)

      # 3. Add bias dimension and transform into columns
      if bias_trick:
        ones_train = torch.ones(X_train.shape[0], 1, device=X_train.device)
        X_train = torch.cat([X_train, ones_train], dim=1)
        ones_test = torch.ones(X_test.shape[0], 1, device=X_test.device)
        X_test = torch.cat([X_test, ones_test], dim=1)
      
      # 4. split train/test
      # For random permumation, you can use torch.randperm or torch.randint
      # But, for this homework, we use slicing instead.
      validation_ratio = 0.2
      num_training = int( X_train.shape[0] * (1.0 - validation_ratio) )
      num_validation = X_train.shape[0] - num_training

      # return the dataset
      data_dict = {}
      data_dict['X_val'] = X_train[num_training:num_training + num_validation]
      data_dict['y_val'] = y_train[num_training:num_training + num_validation]
      data_dict['X_train'] = X_train[0:num_training]
      data_dict['y_train'] = y_train[0:num_training]

      data_dict['X_test'] = X_test
      data_dict['y_test'] = y_test
    return data_dict
