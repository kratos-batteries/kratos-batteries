"""
This module contains all functions required by the user to predict the
volumetric and gravimetric capacity of an electrode material.
"""

import pandas as pd
import numpy as np
import torch
import torch.optim as optim

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def get_predictions(model, x_test):
    """Generates the predicted values for capacity given the trained
       neural network"""

    inputs = np.array(x_test)
    inputs_torch = torch.tensor(inputs, dtype = torch.float)

    return model(inputs_torch)


def convert(y_param):
    """
    Function to convert from scaled data to normal data
    This function takes the output of the model and will return the actual
    property quantities.
    """
    df_output = pd.read_csv('../kratosbat/Data/Training\
        Data.csv')[['Gravimetric Capacity (units)', 'Volumetric Capacity']]
    m_s = MinMaxScaler()
    m_s.fit(df_output)
    result = m_s.inverse_transform(y_param)
    print(result)


def nn_capacity(dataframe, d_in, H, h_2, d_out, N):
    """
    Takes a dataframe, the number of inputs, the number of nodes
    for the first hidden layer, the number of nodes for the second
    hidden layer, the number of outputs, and the total number of
    datapoints as the input parameters and returns a trained
    neural network to return gravimetric and volumetric capacity
    """
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

        return model


def nn_volume(dataframe, d_in, H, h_2, d_out, N):
    """
    Takes dataframe and returns the volume change of the electrode. Parameters
    are the dataframe, the number of inputs, the number of nodes for the
    first hidden layer, the number of nodes for the second hidden layer, the
    number of outputs, and the total number of datapoints.
    """

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

        return model


def generate_model_df(ion, crystal_system, spacegroup, charge, discharge):
    """
    Generates dataframe to input into the neural network model.
    The dataframe must contain the working ion, crystal system,
    spacegroup number, charge formula, and discharge formula
    """

    bat_dataframe = pd.DataFrame()
    bat_dataframe['Working Ion'] = [str(ion)]
    bat_dataframe['Crystal System'] = [str(crystal_system)]
    bat_dataframe['Spacegroup Number'] = [spacegroup]
    bat_dataframe['Charge Formula'] = [str(charge)]
    bat_dataframe['Discharge Formula'] = [str(discharge)]

    from magpie import MagpieServer

    m = MagpieServer()
    charge_formula = bat_dataframe['Charge Formula']
    df_Mean_Charge = m.generate_attributes("oqmd-Eg", charge_formula)\
        .iloc[:, 6:-7:6]
    df_Dev_Charge = m.generate_attributes("oqmd-Eg", charge_formula)\
        .iloc[:, 8:-7:6]
    df_Mean_Charge.rename(
        columns={'mean_Nuber': 'Char_mean_Number',
                 'mean_MendeleevNumber': 'Char_mean_MendeleevNumber',
                 'mean_AtomicWeight': 'Char_mean_AtomicWeight',
                 'mean_MeltingT': 'Char_mean_MeltingTemp',
                 'mean_Column': 'Char_mean_Column',
                 'mean_Row': 'Char_mean_Row',
                 'mean_CovalentRadius': 'Char_mean_CovalentRadius',
                 'mean_Electronegativity': 'Char_mean_Electronegativity',
                 'mean_NsValence': 'Char_mean_NsValence',
                 'mean_NpValence': 'Char_mean_NpValence',
                 'mean_NdValence': 'Char_mean_NdValence',
                 'mean_NfValence': 'Char_mean_NfValence',
                 'mean_NValance': 'Char_mean_NValance',
                 'mean_NsUnfilled': 'Char_mean_NsUnfilled',
                 'mean_NpUnfilled': 'Char_mean_NpUnfilled',
                 'mean_NdUnfilled': 'Char_mean_NdUnfilled',
                 'mean_NfUnfilled': 'Char_mean_NfUnfilled',
                 'mean_NUnfilled': 'Char_mean_NUnfilled',
                 'mean_GSvolume_pa': 'Char_mean_GSvolume_pa',
                 'mean_GSbandgap': 'Char_mean_GSbandgap',
                 'mean_GSmagmom': 'Char_mean_GSmagmom',
                 'mean_SpaceGroupNumber': 'Char_mean_SpaceGroupNumber'})
    df_Dev_Charge.rename(
        columns={'dev_Nuber': 'Char_dev_Number',
                 'dev_MendeleevNumber': 'Char_dev_MendeleevNumber',
                 'dev_AtomicWeight': 'Char_dev_AtomicWeight',
                 'dev_MeltingT': 'Char_dev_MeltingTemp',
                 'dev_Column': 'Char_dev_Column',
                 'dev_Row': 'Char_dev_Row',
                 'dev_CovalentRadius': 'Char_dev_CovalentRadius',
                 'dev_Electronegativity': 'Char_dev_Electronegativity',
                 'dev_NsValence': 'Char_dev_NsValence',
                 'dev_NpValence': 'Char_dev_NpValence',
                 'dev_NdValence': 'Char_dev_NdValence',
                 'dev_NfValence': 'Char_dev_NfValence',
                 'dev_NValance': 'Char_dev_NValance',
                 'dev_NsUnfilled': 'Char_dev_NsUnfilled',
                 'dev_NpUnfilled': 'Char_dev_NpUnfilled',
                 'dev_NdUnfilled': 'Char_dev_NdUnfilled',
                 'dev_NfUnfilled': 'Char_dev_NfUnfilled',
                 'dev_NUnfilled': 'Char_dev_NUnfilled',
                 'dev_GSvolume_pa': 'Char_dev_GSvolume_pa',
                 'dev_GSbandgap': 'Char_dev_GSbandgap',
                 'dev_GSmagmom': 'Char_dev_GSmagmom',
                 'dev_SpaceGroupNumber': 'Char_dev_SpaceGroupNumber'})

    # here we can get Mean and Deviation of Element Property for
    # Discharge_Formula
    discharge_formula = bat_dataframe['Discharge Formula']
    df_Mean_Discharge = m.generate_attributes("oqmd-Eg", discharge_formula)\
        .iloc[:, 6:-7:6]
    df_Dev_Discharge = m.generate_attributes("oqmd-Eg", discharge_formula)\
        .iloc[:, 8:-7:6]
    df_Mean_Discharge.rename(
        columns={'mean_Nuber': 'Dis_mean_Number',
                 'mean_MendeleevNumber': 'Dis_mean_MendeleevNumber',
                 'mean_AtomicWeight': 'Dis_mean_AtomicWeight',
                 'mean_MeltingT': 'Dis_mean_MeltingTemp',
                 'mean_Column': 'Dis_mean_Column',
                 'mean_Row': 'Dis_mean_Row',
                 'mean_CovalentRadius': 'Dis_mean_CovalentRadius',
                 'mean_Electronegativity': 'Dis_mean_Electronegativity',
                 'mean_NsValence': 'Dis_mean_NsValence',
                 'mean_NpValence': 'Dis_mean_NpValence',
                 'mean_NdValence': 'Dis_mean_NdValence',
                 'mean_NfValence': 'Dis_mean_NfValence',
                 'mean_NValance': 'Dis_mean_NValance',
                 'mean_NsUnfilled': 'Dis_mean_NsUnfilled',
                 'mean_NpUnfilled': 'Dis_mean_NpUnfilled',
                 'mean_NdUnfilled': 'Dis_mean_NdUnfilled',
                 'mean_NfUnfilled': 'Dis_mean_NfUnfilled',
                 'mean_NUnfilled': 'Dis_mean_NUnfilled',
                 'mean_GSvolume_pa': 'Dis_mean_GSvolume_pa',
                 'mean_GSbandgap': 'Dis_mean_GSbandgap',
                 'mean_GSmagmom': 'Dis_mean_GSmagmom',
                 'mean_SpaceGroupNumber': 'Dis_mean_SpaceGroupNumber'})
    df_Dev_Discharge.rename(
        columns={'dev_Nuber': 'Dis_dev_Number',
                 'dev_MendeleevNumber': 'Dis_dev_MendeleevNumber',
                 'dev_AtomicWeight': 'Dis_dev_AtomicWeight',
                 'dev_MeltingT': 'Dis_dev_MeltingTemp',
                 'dev_Column': 'Dis_dev_Column',
                 'dev_Row': 'Dis_dev_Row',
                 'dev_CovalentRadius': 'Dis_dev_CovalentRadius',
                 'dev_Electronegativity': 'Dis_dev_Electronegativity',
                 'dev_NsValence': 'Dis_dev_NsValence',
                 'dev_NpValence': 'Dis_dev_NpValence',
                 'dev_NdValence': 'Dis_dev_NdValence',
                 'dev_NfValence': 'Dis_dev_NfValence',
                 'dev_NValance': 'Dis_dev_NValance',
                 'dev_NsUnfilled': 'Dis_dev_NsUnfilled',
                 'dev_NpUnfilled': 'Dis_dev_NpUnfilled',
                 'dev_NdUnfilled': 'Dis_dev_NdUnfilled',
                 'dev_NfUnfilled': 'Dis_dev_NfUnfilled',
                 'dev_NUnfilled': 'Dis_dev_NUnfilled',
                 'dev_GSvolume_pa': 'Dis_dev_GSvolume_pa',
                 'dev_GSbandgap': 'Dis_dev_GSbandgap',
                 'dev_GSmagmom': 'Dis_dev_GSmagmom',
                 'dev_SpaceGroupNumber': 'Dis_dev_SpaceGroupNumber'})

    # use concat to merge all data in one DataFrame
    user_bat = bat_dataframe.drop(
        ['Charge Formula', 'Discharge Formula'], axis=1)
    bat_data_dataframe = pd.concat(
        objs=[user_bat, df_Mean_Charge, df_Dev_Charge,
              df_Mean_Discharge, df_Dev_Discharge], axis=1)

    def pcacsv(DF):

        # Change strings to numbers

        LB = preprocessing.LabelBinarizer()
        WI = LB.fit_transform(np.array(DF.loc[:, ['Working Ion']]))
        CS = LB.fit_transform(np.array(DF.loc[:, ['Crystal System']]))
        SN = LB.fit_transform(np.array(DF.loc[:, ['Spacegroup Number']]))
        EL = np.array(DF.loc[:, ['mean_Number', 'mean_MendeleevNumber',
                                 'mean_AtomicWeight', 'mean_MeltingT',
                                 'mean_Column', 'mean_Row',
                                 'mean_CovalentRadius',
                                 'mean_Electronegativity', 'mean_NsValence',
                                 'mean_NpValence', 'mean_NdValence',
                                 'mean_NfValence', 'mean_NValance',
                                 'mean_NsUnfilled', 'mean_NpUnfilled',
                                 'mean_NdUnfilled', 'mean_NfUnfilled',
                                 'mean_NUnfilled', 'mean_GSvolume_pa',
                                 'mean_GSbandgap', 'mean_GSmagmom',
                                 'mean_SpaceGroupNumber',
                                 'dev_Number', 'dev_MendeleevNumber',
                                 'dev_AtomicWeight', 'dev_MeltingT',
                                 'dev_Column', 'dev_Row', 'dev_CovalentRadius',
                                 'dev_Electronegativity', 'dev_NsValence',
                                 'dev_NpValence', 'dev_NdValence',
                                 'dev_NfValence', 'dev_NValance',
                                 'dev_NsUnfilled', 'dev_NpUnfilled',
                                 'dev_NdUnfilled', 'dev_NfUnfilled',
                                 'dev_NUnfilled', 'dev_GSvolume_pa',
                                 'dev_GSbandgap', 'dev_GSmagmom',
                                 'dev_SpaceGroupNumber', 'mean_Number.1',
                                 'mean_MendeleevNumber.1',
                                 'mean_AtomicWeight.1', 'mean_MeltingT.1',
                                 'mean_Column.1', 'mean_Row.1',
                                 'mean_CovalentRadius.1',
                                 'mean_Electronegativity.1',
                                 'mean_NsValence.1', 'mean_NpValence.1',
                                 'mean_NdValence.1', 'mean_NfValence.1',
                                 'mean_NValance.1', 'mean_NsUnfilled.1',
                                 'mean_NpUnfilled.1', 'mean_NdUnfilled.1',
                                 'mean_NfUnfilled.1', 'mean_NUnfilled.1',
                                 'mean_GSvolume_pa.1', 'mean_GSbandgap.1',
                                 'mean_GSmagmom.1', 'mean_SpaceGroupNumber.1',
                                 'dev_Number.1', 'dev_MendeleevNumber.1',
                                 'dev_AtomicWeight.1', 'dev_MeltingT.1',
                                 'dev_Column.1', 'dev_Row.1',
                                 'dev_CovalentRadius.1',
                                 'dev_Electronegativity.1', 'dev_NsValence.1',
                                 'dev_NpValence.1', 'dev_NdValence.1',
                                 'dev_NfValence.1', 'dev_NValance.1',
                                 'dev_NsUnfilled.1', 'dev_NpUnfilled.1',
                                 'dev_NdUnfilled.1', 'dev_NfUnfilled.1',
                                 'dev_NUnfilled.1', 'dev_GSvolume_pa.1',
                                 'dev_GSbandgap.1', 'dev_GSmagmom.1',
                                 'dev_SpaceGroupNumber.1']])
        PROP = np.hstack((WI, CS, SN, EL))
        return DF, PROP

    DF, PROP = pcacsv(bat_data_dataframe)

    PROP = PROP[np.logical_not(np.isnan(PROP))]
    # Use MinMaxScaler

    MS = MinMaxScaler()
    PMS = MS.fit_transform(PROP.reshape(-1, 1))

    PCA2 = PCA(n_components=1)
    NEW_DATA2 = PCA2.fit_transform(PMS)

    NEW_DF2 = pd.DataFrame(NEW_DATA2)
    df = np.array(NEW_DF2)
    df = np.transpose(df)

    return df
