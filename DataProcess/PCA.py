import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df=pd.read_csv('TrainingData.csv')

#Change strings to numbers
from sklearn import preprocessing

# use LabelBinarizer package in sklearn which can A simple way to extend these algorithms
#  to the multi-class classification case is to use the so-called one-vs-all scheme.
lb = preprocessing.LabelBinarizer()
wi=lb.fit_transform(np.array(df.loc[:,['Working Ion']]))
cs=lb.fit_transform(np.array(df.loc[:,['Crystal System']]))
sn=lb.fit_transform(np.array(df.loc[:,['Spacegroup Number']]))
el=np.array(df.loc[:,['mean_Number', 'mean_MendeleevNumber',
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
prop=np.hstack((wi, cs, sn, el))

#Use StandardScaler, normalize data with minus its mean and divide its variance
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
pss = ss.fit_transform(prop)

pca = PCA(n_components=165)
newdata=pca.fit_transform(pss)

newdf = pd.DataFrame(newdata)
newdf.to_csv('NEWTrainingData_StandardScaler.csv')

#Use MinMaxScaler, put all the magnitude of value in range -1~1
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()
pms = ms.fit_transform(prop)

pca2 = PCA(n_components=115)
newdata2=pca2.fit_transform(pms)

newdf2 = pd.DataFrame(newdata2)
newdf2.to_csv('NEWTrainingData_MinMaxScaler.csv')