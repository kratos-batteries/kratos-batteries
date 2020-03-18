import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def generate_model_df(Ion, CrystalSystem, Spacegroup, Charge, Discharge):
    """Generates dataframe to input into the neural network model. The dataframe must contain
    the working ion, crystal system, spacegroup number, charge formula, and discharge formula"""
    import pandas as pd

    bat_dataframe = pd.DataFrame()
    bat_dataframe['Working Ion'] = [str(Ion)]
    bat_dataframe['Crystal System'] = [str(CrystalSystem)]
    bat_dataframe['Spacegroup Number'] = [Spacegroup]
    bat_dataframe['Charge Formula'] = [str(Charge)]
    bat_dataframe['Discharge Formula'] = [str(Discharge)]

    from magpie import MagpieServer

    m = MagpieServer()
    charge_formula=bat_dataframe['Charge Formula']
    df_Mean_Charge=m.generate_attributes("oqmd-Eg",charge_formula).iloc[:,6:-7:6]
    df_Dev_Charge=m.generate_attributes("oqmd-Eg",charge_formula).iloc[:,8:-7:6]
    df_Mean_Charge.rename(columns={'mean_Nuber':'Char_mean_Number','mean_MendeleevNumber':'Char_mean_MendeleevNumber','mean_AtomicWeight':'Char_mean_AtomicWeight','mean_MeltingT':'Char_mean_MeltingTemp','mean_Column':'Char_mean_Column','mean_Row':'Char_mean_Row',
        'mean_CovalentRadius':'Char_mean_CovalentRadius','mean_Electronegativity':'Char_mean_Electronegativity','mean_NsValence':'Char_mean_NsValence','mean_NpValence':'Char_mean_NpValence','mean_NdValence':'Char_mean_NdValence','mean_NfValence':'Char_mean_NfValence',
        'mean_NValance':'Char_mean_NValance','mean_NsUnfilled':'Char_mean_NsUnfilled','mean_NpUnfilled':'Char_mean_NpUnfilled','mean_NdUnfilled':'Char_mean_NdUnfilled','mean_NfUnfilled':'Char_mean_NfUnfilled','mean_NUnfilled':'Char_mean_NUnfilled','mean_GSvolume_pa':'Char_mean_GSvolume_pa',
        'mean_GSbandgap':'Char_mean_GSbandgap','mean_GSmagmom':'Char_mean_GSmagmom','mean_SpaceGroupNumber':'Char_mean_SpaceGroupNumber'})
    df_Dev_Charge.rename(columns={'dev_Nuber':'Char_dev_Number','dev_MendeleevNumber':'Char_dev_MendeleevNumber','dev_AtomicWeight':'Char_dev_AtomicWeight','dev_MeltingT':'Char_dev_MeltingTemp','dev_Column':'Char_dev_Column','dev_Row':'Char_dev_Row',
        'dev_CovalentRadius':'Char_dev_CovalentRadius','dev_Electronegativity':'Char_dev_Electronegativity','dev_NsValence':'Char_dev_NsValence','dev_NpValence':'Char_dev_NpValence','dev_NdValence':'Char_dev_NdValence','dev_NfValence':'Char_dev_NfValence',
        'dev_NValance':'Char_dev_NValance','dev_NsUnfilled':'Char_dev_NsUnfilled','dev_NpUnfilled':'Char_dev_NpUnfilled','dev_NdUnfilled':'Char_dev_NdUnfilled','dev_NfUnfilled':'Char_dev_NfUnfilled','dev_NUnfilled':'Char_dev_NUnfilled','dev_GSvolume_pa':'Char_dev_GSvolume_pa',
        'dev_GSbandgap':'Char_dev_GSbandgap','dev_GSmagmom':'Char_dev_GSmagmom','dev_SpaceGroupNumber':'Char_dev_SpaceGroupNumber'})


        # here we can get Mean and Deviation of Element Property for Discharge_Formula
    discharge_formula=bat_dataframe['Discharge Formula']
    df_Mean_Discharge=m.generate_attributes("oqmd-Eg",discharge_formula).iloc[:,6:-7:6]
    df_Dev_Discharge=m.generate_attributes("oqmd-Eg",discharge_formula).iloc[:,8:-7:6]
    df_Mean_Discharge.rename(columns={'mean_Nuber':'Dis_mean_Number','mean_MendeleevNumber':'Dis_mean_MendeleevNumber','mean_AtomicWeight':'Dis_mean_AtomicWeight','mean_MeltingT':'Dis_mean_MeltingTemp','mean_Column':'Dis_mean_Column','mean_Row':'Dis_mean_Row',
        'mean_CovalentRadius':'Dis_mean_CovalentRadius','mean_Electronegativity':'Dis_mean_Electronegativity','mean_NsValence':'Dis_mean_NsValence','mean_NpValence':'Dis_mean_NpValence','mean_NdValence':'Dis_mean_NdValence','mean_NfValence':'Dis_mean_NfValence',
        'mean_NValance':'Dis_mean_NValance','mean_NsUnfilled':'Dis_mean_NsUnfilled','mean_NpUnfilled':'Dis_mean_NpUnfilled','mean_NdUnfilled':'Dis_mean_NdUnfilled','mean_NfUnfilled':'Dis_mean_NfUnfilled','mean_NUnfilled':'Dis_mean_NUnfilled','mean_GSvolume_pa':'Dis_mean_GSvolume_pa',
        'mean_GSbandgap':'Dis_mean_GSbandgap','mean_GSmagmom':'Dis_mean_GSmagmom','mean_SpaceGroupNumber':'Dis_mean_SpaceGroupNumber'})
    df_Dev_Discharge.rename(columns={'dev_Nuber':'Dis_dev_Number','dev_MendeleevNumber':'Dis_dev_MendeleevNumber','dev_AtomicWeight':'Dis_dev_AtomicWeight','dev_MeltingT':'Dis_dev_MeltingTemp','dev_Column':'Dis_dev_Column','dev_Row':'Dis_dev_Row',
        'dev_CovalentRadius':'Dis_dev_CovalentRadius','dev_Electronegativity':'Dis_dev_Electronegativity','dev_NsValence':'Dis_dev_NsValence','dev_NpValence':'Dis_dev_NpValence','dev_NdValence':'Dis_dev_NdValence','dev_NfValence':'Dis_dev_NfValence',
        'dev_NValance':'Dis_dev_NValance','dev_NsUnfilled':'Dis_dev_NsUnfilled','dev_NpUnfilled':'Dis_dev_NpUnfilled','dev_NdUnfilled':'Dis_dev_NdUnfilled','dev_NfUnfilled':'Dis_dev_NfUnfilled','dev_NUnfilled':'Dis_dev_NUnfilled','dev_GSvolume_pa':'Dis_dev_GSvolume_pa',
        'dev_GSbandgap':'Dis_dev_GSbandgap','dev_GSmagmom':'Dis_dev_GSmagmom','dev_SpaceGroupNumber':'Dis_dev_SpaceGroupNumber'})

    # use concat to merge all data in one DataFrame
    user_bat = bat_dataframe.drop(['Charge Formula', 'Discharge Formula'], axis=1)
    bat_data_dataframe=pd.concat(objs=[user_bat,df_Mean_Charge,df_Dev_Charge,df_Mean_Discharge,df_Dev_Discharge],axis=1)

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

    DF, PROP = pcacsv(bat_data_dataframe)

    PROP = PROP[np.logical_not(np.isnan(PROP))]
    #Use MinMaxScaler

    MS = MinMaxScaler()
    PMS = MS.fit_transform(PROP.reshape(-1,1))

    PCA2 = PCA(n_components=1)
    NEW_DATA2 = PCA2.fit_transform(PMS)

    NEW_DF2 = pd.DataFrame(NEW_DATA2)

    return NEW_DF2
