import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

#categorizing the input and output data
bat_data = pd.read_csv('TrainingData.csv')
train_bat = bat_data

train_bat = train_bat.replace('Li', 1)
train_bat = train_bat.replace('Ca', 2)
train_bat = train_bat.replace('Cs', 3)
train_bat = train_bat.replace('Rb', 4)
train_bat = train_bat.replace('K', 5)
train_bat = train_bat.replace('Y', 6)
train_bat = train_bat.replace('Na', 7)
train_bat = train_bat.replace('Al', 8)
train_bat = train_bat.replace('Zn', 9)
train_bat = train_bat.replace('Mg', 0)

train_bat = train_bat.replace('Orthorombic', 1)
train_bat = train_bat.replace('Monoclinic', 2)
train_bat = train_bat.replace('Trigonal', 3)
train_bat = train_bat.replace('Triclinic', 4)
train_bat = train_bat.replace('Tetragonal', 5)
train_bat = train_bat.replace('Hexagonal', 6)
train_bat = train_bat.replace('Cubic', 7)

D_in, H, H2, D_out, N = 91, 1000, 75, 3, 4000
dtype = torch.float
device = torch.device('cpu')


#define testing sets
x_train = train_bat
x_train = x_train.drop(columns=['Battery ID', 'Gravimetric Capacity (units)', 'Volumetric Capacity', 'Max Delta Volume'])

y_train = train_bat[['Gravimetric Capacity (units)', 'Volumetric Capacity', 'Max Delta Volume']]

x_test = x_train[4000:]
y_test = y_train[4000:]

x_train = x_train[:4000]
y_train = y_train[:4000]

#Defining training and testing data
x_train_np = np.array(x_train)
x_train_torch = torch.tensor(x_train_np, device = device, dtype = dtype)

y_train_np = np.array(y_train)
y_train_torch = torch.tensor(y_train_np, device = device, dtype = dtype)

x_test_np = np.array(x_test)
x_test_torch = torch.tensor(x_test_np, device = device, dtype = dtype)

y_test_np = np.array(y_test)
y_test_torch = torch.tensor(y_test_np, device = device, dtype = dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

#define model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H, H2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H2, D_out),
)

learning_rate = 1e-10
for t in range(10000):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x_train_torch)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y_train_torch)
    if t % 1000 == 999:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
