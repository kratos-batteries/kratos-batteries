"""
This PCA analysis will produce normalized data to \
be used for the training model
"""

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def pcacsv(DF):

    #Change strings to numbers

    LB = preprocessing.LabelBinarizer()
    WI=LB.fit_transform(np.array(DF.loc[:,['Working Ion']]))
    CS=LB.fit_transform(np.array(DF.loc[:,['Crystal System']]))
    SN=LB.fit_transform(np.array(DF.loc[:,['Spacegroup Number']]))
    EL=np.array(DF.loc[:,['mean_Number', 'mean_MendeleevNumber',
                       'mean_AtomicWeight', 'mean_MeltingT', 'mean_Column', 'mean_Row',
                       'mean_CovalentRadius', 'mean_Electronegativity', 'mean_NsValence',
                       'mean_NpValence', 'mean_NdValence', 'mean_NfValence', 'mean_NValance',
                       'mean_NsUnfilled', 'mean_NpUnfilled', 'mean_NdUnfilled',
                       'mean_NfUnfilled', 'mean_NUnfilled', 'mean_GSvolume_pa',
                       'mean_GSbandgap', 'mean_GSmagmom', 'mean_SpaceGroupNumber',
                       'dev_Number', 'dev_MendeleevNumber', 'dev_AtomicWeight', 'dev_MeltingT',
                       'dev_Column', 'dev_Row', 'dev_CovalentRadius', 'dev_Electronegativity',
                       'dev_NsValence', 'dev_NpValence', 'dev_NdValence', 'dev_NfValence',
                       'dev_NValance', 'dev_NsUnfilled', 'dev_NpUnfilled', 'dev_NdUnfilled',
                       'dev_NfUnfilled', 'dev_NUnfilled', 'dev_GSvolume_pa', 'dev_GSbandgap',
                       'dev_GSmagmom', 'dev_SpaceGroupNumber', 'mean_Number.1',
                       'mean_MendeleevNumber.1', 'mean_AtomicWeight.1', 'mean_MeltingT.1',
                       'mean_Column.1', 'mean_Row.1', 'mean_CovalentRadius.1',
                       'mean_Electronegativity.1', 'mean_NsValence.1', 'mean_NpValence.1',
                       'mean_NdValence.1', 'mean_NfValence.1', 'mean_NValance.1',
                       'mean_NsUnfilled.1', 'mean_NpUnfilled.1', 'mean_NdUnfilled.1',
                       'mean_NfUnfilled.1', 'mean_NUnfilled.1', 'mean_GSvolume_pa.1',
                       'mean_GSbandgap.1', 'mean_GSmagmom.1', 'mean_SpaceGroupNumber.1',
                       'dev_Number.1', 'dev_MendeleevNumber.1', 'dev_AtomicWeight.1',
                       'dev_MeltingT.1', 'dev_Column.1', 'dev_Row.1', 'dev_CovalentRadius.1',
                       'dev_Electronegativity.1', 'dev_NsValence.1', 'dev_NpValence.1',
                       'dev_NdValence.1', 'dev_NfValence.1', 'dev_NValance.1',
                       'dev_NsUnfilled.1', 'dev_NpUnfilled.1', 'dev_NdUnfilled.1',
                       'dev_NfUnfilled.1', 'dev_NUnfilled.1', 'dev_GSvolume_pa.1',
                       'dev_GSbandgap.1', 'dev_GSmagmom.1', 'dev_SpaceGroupNumber.1']])
    PROP = np.hstack((WI, CS, SN, EL))    
    return DF, PROP

def pcasts():
    #Use StandardScaler
    DF, PROP = pcacsv(pd.read_csv('../Data/TrainingData.csv'))

    SS = StandardScaler()
    
    SS.fit(PROP)    
    PSS = SS.transform(PROP)
    
    PCA1 = PCA(n_components=165)
    NEW_DATA = PCA1.fit_transform(PSS)

    NEW_DF = pd.DataFrame(NEW_DATA)
    OUTPUTS = np.array(DF.loc[:, ['Gravimetric Capacity (units)', \
    'Volumetric Capacity', 'Max Delta Volume']])
    NEW_DF['Gravimetric Capacity (units)'] = SS.fit_transform(OUTPUTS)[:, [0]]
    NEW_DF['Volumetric Capacity'] = SS.fit_transform(OUTPUTS)[:, [1]]
    NEW_DF['Max Delta Volume'] = SS.fit_transform(OUTPUTS)[:, [2]]
    NEW_DF.to_csv('../Data/NEWTrainingData_StandardScaler.csv')

    return

def pcamms():
    #Use MinMaxScaler
    DF, PROP = pcacsv(pd.read_csv('../Data/TrainingData.csv'))
    MS = MinMaxScaler()
    PMS = MS.fit_transform(PROP)

    PCA2 = PCA(n_components=115)
    NEW_DATA2 = PCA2.fit_transform(PMS)

    NEW_DF2 = pd.DataFrame(NEW_DATA2)
    OUTPUTS = np.array(DF.loc[:, ['Gravimetric Capacity (units)', \
    'Volumetric Capacity', 'Max Delta Volume']])
    NEW_DF2['Gravimetric Capacity (units)'] = MS.fit_transform(OUTPUTS)[:, [0]]
    NEW_DF2['Volumetric Capacity'] = MS.fit_transform(OUTPUTS)[:, [1]]
    NEW_DF2['Max Delta Volume'] = MS.fit_transform(OUTPUTS)[:, [2]]
    NEW_DF2.to_csv('../Data/NEWTrainingData_MinMaxScaler.csv')

    return

def transts(data):
    #Transform data
    #Use StandardScaler
    CSV = pd.read_csv('../Data/TrainingData.csv')
    DF, PROP = pcacsv(CSV)
    data.columns=['Battery ID', 'Working Ion', 'Crystal System', 'Spacegroup Number',
       'Gravimetric Capacity (units)', 'Volumetric Capacity',
       'Max Delta Volume', 'mean_Number', 'mean_MendeleevNumber',
       'mean_AtomicWeight', 'mean_MeltingT', 'mean_Column', 'mean_Row',
       'mean_CovalentRadius', 'mean_Electronegativity', 'mean_NsValence',
       'mean_NpValence', 'mean_NdValence', 'mean_NfValence', 'mean_NValance',
       'mean_NsUnfilled', 'mean_NpUnfilled', 'mean_NdUnfilled',
       'mean_NfUnfilled', 'mean_NUnfilled', 'mean_GSvolume_pa',
       'mean_GSbandgap', 'mean_GSmagmom', 'mean_SpaceGroupNumber',
       'dev_Number', 'dev_MendeleevNumber', 'dev_AtomicWeight', 'dev_MeltingT',
       'dev_Column', 'dev_Row', 'dev_CovalentRadius', 'dev_Electronegativity',
       'dev_NsValence', 'dev_NpValence', 'dev_NdValence', 'dev_NfValence',
       'dev_NValance', 'dev_NsUnfilled', 'dev_NpUnfilled', 'dev_NdUnfilled',
       'dev_NfUnfilled', 'dev_NUnfilled', 'dev_GSvolume_pa', 'dev_GSbandgap',
       'dev_GSmagmom', 'dev_SpaceGroupNumber', 'mean_Number.1',
       'mean_MendeleevNumber.1', 'mean_AtomicWeight.1', 'mean_MeltingT.1',
       'mean_Column.1', 'mean_Row.1', 'mean_CovalentRadius.1',
       'mean_Electronegativity.1', 'mean_NsValence.1', 'mean_NpValence.1',
       'mean_NdValence.1', 'mean_NfValence.1', 'mean_NValance.1',
       'mean_NsUnfilled.1', 'mean_NpUnfilled.1', 'mean_NdUnfilled.1',
       'mean_NfUnfilled.1', 'mean_NUnfilled.1', 'mean_GSvolume_pa.1',
       'mean_GSbandgap.1', 'mean_GSmagmom.1', 'mean_SpaceGroupNumber.1',
       'dev_Number.1', 'dev_MendeleevNumber.1', 'dev_AtomicWeight.1',
       'dev_MeltingT.1', 'dev_Column.1', 'dev_Row.1', 'dev_CovalentRadius.1',
       'dev_Electronegativity.1', 'dev_NsValence.1', 'dev_NpValence.1',
       'dev_NdValence.1', 'dev_NfValence.1', 'dev_NValance.1',
       'dev_NsUnfilled.1', 'dev_NpUnfilled.1', 'dev_NdUnfilled.1',
       'dev_NfUnfilled.1', 'dev_NUnfilled.1', 'dev_GSvolume_pa.1',
       'dev_GSbandgap.1', 'dev_GSmagmom.1', 'dev_SpaceGroupNumber.1']
    NData = pd.concat(objs=[data,CSV],axis=0)
    NDF, NPROP = pcacsv(NData)
    
    SS = StandardScaler()
    
    SS.fit(PROP)    
    PSS = SS.transform(PROP)
    
    PCA1 = PCA(n_components=165)
    PCA1.fit(PSS)
    
    NEW_DATA1=PCA1.transform(SS.transform(NPROP[0].reshape(1,-1)))
    return NEW_DATA1

def tranmms(data):
    #Transform data
    #Use MinMaxScaler
    CSV = pd.read_csv('../Data/TrainingData.csv')
    DF, PROP = pcacsv(CSV)
    data.columns=['Battery ID', 'Working Ion', 'Crystal System', 'Spacegroup Number',
       'Gravimetric Capacity (units)', 'Volumetric Capacity',
       'Max Delta Volume', 'mean_Number', 'mean_MendeleevNumber',
       'mean_AtomicWeight', 'mean_MeltingT', 'mean_Column', 'mean_Row',
       'mean_CovalentRadius', 'mean_Electronegativity', 'mean_NsValence',
       'mean_NpValence', 'mean_NdValence', 'mean_NfValence', 'mean_NValance',
       'mean_NsUnfilled', 'mean_NpUnfilled', 'mean_NdUnfilled',
       'mean_NfUnfilled', 'mean_NUnfilled', 'mean_GSvolume_pa',
       'mean_GSbandgap', 'mean_GSmagmom', 'mean_SpaceGroupNumber',
       'dev_Number', 'dev_MendeleevNumber', 'dev_AtomicWeight', 'dev_MeltingT',
       'dev_Column', 'dev_Row', 'dev_CovalentRadius', 'dev_Electronegativity',
       'dev_NsValence', 'dev_NpValence', 'dev_NdValence', 'dev_NfValence',
       'dev_NValance', 'dev_NsUnfilled', 'dev_NpUnfilled', 'dev_NdUnfilled',
       'dev_NfUnfilled', 'dev_NUnfilled', 'dev_GSvolume_pa', 'dev_GSbandgap',
       'dev_GSmagmom', 'dev_SpaceGroupNumber', 'mean_Number.1',
       'mean_MendeleevNumber.1', 'mean_AtomicWeight.1', 'mean_MeltingT.1',
       'mean_Column.1', 'mean_Row.1', 'mean_CovalentRadius.1',
       'mean_Electronegativity.1', 'mean_NsValence.1', 'mean_NpValence.1',
       'mean_NdValence.1', 'mean_NfValence.1', 'mean_NValance.1',
       'mean_NsUnfilled.1', 'mean_NpUnfilled.1', 'mean_NdUnfilled.1',
       'mean_NfUnfilled.1', 'mean_NUnfilled.1', 'mean_GSvolume_pa.1',
       'mean_GSbandgap.1', 'mean_GSmagmom.1', 'mean_SpaceGroupNumber.1',
       'dev_Number.1', 'dev_MendeleevNumber.1', 'dev_AtomicWeight.1',
       'dev_MeltingT.1', 'dev_Column.1', 'dev_Row.1', 'dev_CovalentRadius.1',
       'dev_Electronegativity.1', 'dev_NsValence.1', 'dev_NpValence.1',
       'dev_NdValence.1', 'dev_NfValence.1', 'dev_NValance.1',
       'dev_NsUnfilled.1', 'dev_NpUnfilled.1', 'dev_NdUnfilled.1',
       'dev_NfUnfilled.1', 'dev_NUnfilled.1', 'dev_GSvolume_pa.1',
       'dev_GSbandgap.1', 'dev_GSmagmom.1', 'dev_SpaceGroupNumber.1']
    NData = pd.concat(objs=[data,CSV],axis=0)
    NDF, NPROP = pcacsv(NData)

    
    MS = MinMaxScaler()
    
    MS.fit(PROP)    
    PMS = MS.transform(PROP)
    
    PCA2 = PCA(n_components=115)
    PCA2.fit(PMS)
    
    NEW_DATA2=PCA2.transform(MS.transform(NPROP[0].reshape(1,-1)))
    return NEW_DATA2