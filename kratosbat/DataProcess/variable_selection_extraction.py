"""
This sub module's purpose is to obtain a new normalized \
dataset for SVR
"""

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def get_data_rfe():
    """
    from this function we can get a data set to train our RFE model to select variable
    """
    trainset = pd.read_csv('TrainingData.csv')
    lb_model = preprocessing.LabelBinarizer()
    wi_data = lb_model.fit_transform(np.array(trainset.loc[:, ['Working Ion']]))
    cs_data = lb_model.fit_transform(np.array(trainset.loc[:, ['Crystal System']]))
    sn_data = lb_model.fit_transform(np.array(trainset.loc[:, ['Spacegroup Number']]))
    el_data = np.array(trainset.loc[:, ['mean_Number', 'mean_MendeleevNumber',
                                        'mean_AtomicWeight', 'mean_MeltingT',
                                        'mean_Column', 'mean_Row',
                                        'mean_CovalentRadius', 'mean_Electronegativity',
                                        'mean_NsValence', 'mean_NpValence', 'mean_NdValence',
                                        'mean_NfValence', 'mean_NValance', 'mean_NsUnfilled',
                                        'mean_NpUnfilled', 'mean_NdUnfilled', 'mean_NfUnfilled',
                                        'mean_NUnfilled', 'mean_GSvolume_pa', 'mean_GSbandgap',
                                        'mean_GSmagmom', 'mean_SpaceGroupNumber', 'dev_Number',
                                        'dev_MendeleevNumber', 'dev_AtomicWeight', 'dev_MeltingT',
                                        'dev_Column', 'dev_Row', 'dev_CovalentRadius',
                                        'dev_Electronegativity', 'dev_NsValence',
                                        'dev_NpValence', 'dev_NdValence', 'dev_NfValence',
                                        'dev_NValance', 'dev_NsUnfilled', 'dev_NpUnfilled',
                                        'dev_NdUnfilled', 'dev_NfUnfilled', 'dev_NUnfilled',
                                        'dev_GSvolume_pa', 'dev_GSbandgap', 'dev_GSmagmom',
                                        'dev_SpaceGroupNumber', 'mean_Number.1',
                                        'mean_MendeleevNumber.1', 'mean_AtomicWeight.1',
                                        'mean_MeltingT.1', 'mean_Column.1', 'mean_Row.1',
                                        'mean_CovalentRadius.1', 'mean_Electronegativity.1',
                                        'mean_NsValence.1', 'mean_NpValence.1', 'mean_NdValence.1',
                                        'mean_NfValence.1', 'mean_NValance.1', 'mean_NsUnfilled.1',
                                        'mean_NpUnfilled.1', 'mean_NdUnfilled.1',
                                        'mean_NfUnfilled.1', 'mean_NUnfilled.1',
                                        'mean_GSvolume_pa.1', 'mean_GSbandgap.1',
                                        'mean_GSmagmom.1', 'mean_SpaceGroupNumber.1',
                                        'dev_Number.1', 'dev_MendeleevNumber.1',
                                        'dev_AtomicWeight.1', 'dev_MeltingT.1',
                                        'dev_Column.1', 'dev_Row.1', 'dev_CovalentRadius.1',
                                        'dev_Electronegativity.1', 'dev_NsValence.1',
                                        'dev_NpValence.1', 'dev_NdValence.1',
                                        'dev_NfValence.1', 'dev_NValance.1', 'dev_NsUnfilled.1',
                                        'dev_NpUnfilled.1', 'dev_NdUnfilled.1',
                                        'dev_NfUnfilled.1', 'dev_NUnfilled.1',
                                        'dev_GSvolume_pa.1', 'dev_GSbandgap.1',
                                        'dev_GSmagmom.1', 'dev_SpaceGroupNumber.1']])
    prop_data = np.hstack((wi_data, cs_data, sn_data, el_data))
    ss_model = StandardScaler()
    pss_fit = ss_model.fit_transform(prop_data)
    standard_data = pd.DataFrame(pss_fit)

    output_data = \
    pd.read_csv('NEWTrainingData_StandardScaler.csv').loc[:, ['Gravimetric Capacity (units)',
                                                              'Volumetric Capacity',
                                                              'Max Delta Volume']]

    x_train, x_test, y_train, y_test = train_test_split(standard_data,
                                                        output_data,
                                                        test_size=0.2,
                                                        random_state=0)
    return x_train, x_test, y_train, y_test


def svr_linear_rfe(x_train, y_train):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.\
    feature_selection.RFE.html#sklearn.feature_selection.RFE
    a coef_ attribute or a feature_importances_ attribute
    """

    # get a dataframe for Gravimetric Capacity (units) after variable selection
    select1 = RFE(SVR(kernel='linear', degree=3, gamma='scale', \
                      coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, \
                      max_iter=-1), n_features_to_select=208)
    gc_df = pd.DataFrame(select1.fit_transform(x_train, y_train['Gravimetric Capacity (units)']))

    # get a dataframe for Volumetric Capacity after variable selection
    select2 = RFE(SVR(kernel='linear', degree=3, gamma='scale', \
                      coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, \
                      max_iter=-1), n_features_to_select=207)
    vc_df = pd.DataFrame(select2.fit_transform(x_train, y_train['Volumetric Capacity']))

    # get a dataframe for Max Delta Volume after variable selection
    select3 = RFE(SVR(kernel='linear', degree=3, gamma='scale', \
                      coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, \
                      max_iter=-1), n_features_to_select=205)
    mdv_df = pd.DataFrame(select3.fit_transform(x_train, y_train['Max Delta Volume']))

    return gc_df, vc_df, mdv_df


def pca_get_csv(gc_df, vc_df, mdv_df):
    """
    Creating .csv data from new PCA data
    """

    # get a .csv file for Gravimetric Capacity (units) after PCA
    pca_model = PCA(n_components=135)
    new_data1 = pca_model.fit_transform(gc_df)
    new_df1 = pd.DataFrame(new_data1)
    new_df1.to_csv('./Data/Data for svr/GC_PCA.csv')

    # get a .csv file for Volumetric Capacity after PCA
    pca_model = PCA(n_components=135)
    new_data2 = pca_model.fit_transform(vc_df)
    new_df2 = pd.DataFrame(new_data2)
    new_df2.to_csv('./Data/Data for svr/VC_PCA.csv')

    # get a .csv file for Max Delta Volume after PCA
    pca_model = PCA(n_components=131)
    new_data3 = pca_model.fit_transform(mdv_df)
    new_df3 = pd.DataFrame(new_data3)
    new_df3.to_csv('./Data/Data for svr/MDV_PCA.csv')
