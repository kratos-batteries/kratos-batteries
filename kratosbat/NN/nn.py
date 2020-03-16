"""Module that contains the functions responsible for creating trained
neural network models for capacity and for volume change"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd


def nn_capacity(dataframe, d_in, H, h_2, d_out, N):
    """Takes a dataframe, the number of inputs, the number of nodes\
    for the first hidden layer, the number of nodes for the second\
    hidden layer, the number of outputs, and the total number of\
    datapoints as the input parameters and returns a trained\
    neural network to return gravimetric and volumetric capacity"""
    # categorizing the input and output data
    bat_data = pd.read_csv(str(dataframe))
    train_bat = bat_data

    dtype = torch.float
    device = torch.device('cpu')

    # define testing sets
    x_train = train_bat.drop(columns=['Unnamed: 0',
                                      'Gravimetric Capacity (units)',
                                      'Volumetric Capacity',
                                      'Max Delta Volume'])
    y_train = train_bat[['Gravimetric Capacity (units)',
                         'Volumetric Capacity']]

    # shuffle data
    x_train = x_train.sample(frac=1)
    y_train = y_train.sample(frac=1)

    # test-train split
    # optimize for user data

    x_train = x_train[:4000]
    y_train = y_train[:4000]

    # Defining training and testing data
    x_train_np = np.array(x_train)
    x_train_torch = torch.tensor(x_train_np, device=device, dtype=dtype)

    y_train_np = np.array(y_train)
    y_train_torch = torch.tensor(y_train_np, device=device, dtype=dtype)

    # Defining weights
    w1 = torch.randn(d_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, d_out, device=device, dtype=dtype)
    # w3 = torch.randn(H2, D_out, device=device, dtype=dtype)

    # define model
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, H),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(H, h_2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(h_2, d_out),
        # nn.Softmax(dim=1) #normalizing the data,
    )

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    for t in range(1000):
        # Forward pass: compute predicted y by passing
        # x to the model. Module objects
        # override the __call__ operator so you can
        # call them like functions. When
        # doing so you pass Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x_train_torch)

        # Compute and print loss. We pass
        # Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y_train_torch)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero the gradients before running the backward/training pass.
        optimizer.zero_grad()
        model.zero_grad()

        # Backward pass: compute gradient of the
        # loss with respect to all the learnable
        # parameters of the model. Internally,
        # the parameters of each Module are stored
        # in Tensors with requires_grad=True,
        # so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()
        optimizer.step()

        # Update the weights using gradient descent.
        # Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        return(model)


def nn_volume(dataframe, d_in, H, h_2, d_out, N):
    """Takes dataframe and returns the volume change of the\
    electrode. Parameters are the dataframe,\
    the number of inputs, the number of nodes\
    for the first hidden layer, the number of nodes for the second\
    hidden layer, the number of outputs, and the total number of\
    datapoints."""

    # categorizing the input and output data
    bat_data = pd.read_csv(str(dataframe))
    train_bat = bat_data

    dtype = torch.float
    device = torch.device('cpu')

    # define testing sets
    x_train = train_bat[['Gravimetric Capacity (units)',
                         'Volumetric Capacity']]
    y_train = train_bat['Max Delta Volume']

    # shuffle data
    x_train = x_train.sample(frac=1)
    y_train = y_train.sample(frac=1)

    # test-train split
    # optimize for user data

    x_train = x_train[:4000]
    y_train = y_train[:4000]

    # Defining training and testing data
    x_train_np = np.array(x_train)
    x_train_torch = torch.tensor(x_train_np, device=device, dtype=dtype)

    y_train_np = np.array(y_train)
    y_train_torch = torch.tensor(y_train_np, device=device, dtype=dtype)

    # Defining weights
    w1 = torch.randn(d_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, d_out, device=device, dtype=dtype)
    # w3 = torch.randn(H2, D_out, device=device, dtype=dtype)

    # define model
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, H),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(H, h_2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(h_2, d_out),
        # nn.Softmax(dim=1) #normalizing the data,
    )

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    for t in range(1000):
        # Forward pass: compute predicted y by passing
        # x to the model. Module objects
        # override the __call__ operator so you can
        # call them like functions. When
        # doing so you pass Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x_train_torch)

        # Compute and print loss. We pass
        # Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y_train_torch)

        # Zero the gradients before running the backward/training pass.
        optimizer.zero_grad()
        model.zero_grad()

        # Backward pass: compute gradient of the
        # loss with respect to all the learnable
        # parameters of the model. Internally,
        # the parameters of each Module are stored
        # in Tensors with requires_grad=True,
        # so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()
        optimizer.step()

        # Update the weights using gradient descent.
        # Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        return(model)
