import numpy as np
import os
import pandas as pd

from magpie import MagpieServer
from pymatgen import MPRester
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def get_bat_dat(mapi_key):
    """
    Takes materials project API key as input and returns dataframe of
    all battery material properies
    This function returns a dataframe of all the battery materials by cycling
    through each battery ID
    """

    # MAPI_KEY is the API key obtained from the materials project
    mpr = MPRester(mapi_key)
    return

def get_battery_data(self, formula_or_batt_id):
        """
        Returns batteries from a batt id or formula.
        Examples:
            get_battery("mp-300585433")
            get_battery("LiFePO4")
        """
        return mpr._make_request('/battery/%s' % formula_or_batt_id)

def update_check(MAPI_KEY):
    """
    This function tests to see if BatteryData.csv is up to date.
    The function checks the length of the BatteryData.csv and compares it to
    the length of the list produced by getting all of the battery IDs. If the
    lengths are the same, it returns that the data is up to date. If the
    lengths are not the same, it recommends to run get_bat_dat() to update the
    csv file.
    """
    current_df = pd.read_csv('BatteryData.csv')
    current_len = len(current_df)

    # if __name__ == "__main__":
    mpr = MPRester(MAPI_KEY)

    all_bat_ids_list = (mpr._make_request('/battery/all_ids'))

    matproj_len = len(all_bat_ids_list)

    if current_len == matproj_len:
        print("The Current BatteryData.csv file is up to date!")

    else:
        print("Data is not up to date. Please run get_bat_dat() \
               function to obtain new .csv file.")

    return

def get_elementProperty(clean_battery_df=None):

        # here we import a API called The Materials Agnostic Platform for Informatics and Exploration (Magpie)
        # this API can let us use formula of a conpound to get its  elemental properties from statistics of
        # atomic constituents attributes.

        #  Details are in this paper:
        #  Ward, L.; Agrawal, A.; Choudhary, A.; Wolverton, C. A General-Purpose Machine Learning Framework for Predicting Properties of 
        #  Inorganic Materials. npj Comput. Mater. 2016, 2, No. 16028.
        m = MagpieServer()
        if clean_battery_df == None:
            if os.path.exists('./Data/BatteryData.csv'):
                clean_battery_df=pd.read_csv('./Data/BatteryData.csv')
        # here we can get Mean and Deviation of Element Property for Charge_Formula
        charge_formula=clean_battery_df['Charge Formula']
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
        discharge_formula=clean_battery_df['Discharge Formula']
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
        element_attributes=pd.concat(objs=[df_Mean_Charge,df_Dev_Charge,df_Mean_Discharge,df_Dev_Discharge],axis=1)
        element_attributes.to_csv(path_or_buf='./Data/ElementalProperty.csv')
        return element_attributes

def get_all_variable(clean_battery_df=None,element_attributes=None):

     #features(predictor) we are to use: 'Working Ion','Crystal Lattice','Spacegroup', 'element_attribute for charge formula', 'element attributes for discharge fromula'
     # lable to be predict: 'Gravimetric Capacity (units)','Volumetric Capacity','Max Delta Volume'
     if clean_battery_df == None:
         if os.path.exists('./Data/BatteryData.csv'):
             clean_battery_df=pd.read_csv('./Data/BatteryData.csv')
     if element_attributes == None:
         if os.path.exists('./Data/ElementalProperty.csv'):
             element_attributes=pd.read_csv('./Data/ElementalProperty.csv')
     # select features we need in training our model
     train_set=clean_battery_df[['Working Ion','Crystal Lattice','Spacegroup Number','Gravimetric Capacity (units)','Volumetric Capacity','Max Delta Volume']]
     # concat the element attributes which represent the property of charge/dis electrode in our features
     train_set.reset_index(drop=False, inplace=True)
     train_set=pd.concat(objs=[train_set,element_attributes],axis=1)
     # make a .csv file to our working directory.
     train_set.to_csv(path_or_buf='./Data/TrainingData.csv')
     return train_set

def get_data_RFE():
    # from this function we can get a data set to train our RFE model to select variable
     trainset = pd.read_csv('TrainingData.csv')
     lb = preprocessing.LabelBinarizer()
     wi=lb.fit_transform(np.array(trainset.loc[:,['Working Ion']]))
     cs=lb.fit_transform(np.array(trainset.loc[:,['Crystal System']]))
     sn=lb.fit_transform(np.array(trainset.loc[:,['Spacegroup Number']]))
     el=np.array(trainset.loc[:,['mean_Number', 'mean_MendeleevNumber',
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
     ss = StandardScaler()
     pss = ss.fit_transform(prop)
     standard_data = pd.DataFrame(pss)

     outputs=pd.read_csv('NEWTrainingData_StandardScaler.csv').loc[:,['Gravimetric Capacity (units)', 'Volumetric Capacity', 'Max Delta Volume']]

     X_train,X_test, y_train, y_test =train_test_split(standard_data,outputs,test_size=0.2, random_state=0)
     return X_train,X_test, y_train, y_test

def SVR_linear_RFE(X_train,X_test, y_train, y_test):

    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
    # a coef_ attribute or a feature_importances_ attribute

    # get a dataframe for Gravimetric Capacity (units) after variable selection
    select1 = RFE(SVR(kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1), n_features_to_select=208)
    GC_df = pd.DataFrame(select1.fit_transform(X_train, y_train['Gravimetric Capacity (units)']))

    # get a dataframe for Volumetric Capacity after variable selection
    select2 = RFE(SVR(kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1), n_features_to_select=207)
    VC_df = pd.DataFrame(select2.fit_transform(X_train, y_train['Volumetric Capacity']))

    # get a dataframe for Max Delta Volume after variable selection
    select3 = RFE(SVR(kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1), n_features_to_select=205)
    MDV_df = pd.DataFrame(select3.fit_transform(X_train, y_train['Max Delta Volume']))

    return GC_df, VC_df, MDV_df

def PCA_get_CSV(GC_df, VC_df, MDV_df):

    # get a .csv file for Gravimetric Capacity (units) after PCA
    pca = PCA(n_components=133)
    newdata1=pca.fit_transform(GC_df)
    newdf1 = pd.DataFrame(newdata1)
    newdf1.to_csv('./Data/Data for svr/GC_CPA.csv')

    # get a .csv file for Volumetric Capacity after PCA
    pca = PCA(n_components=134)
    newdata2=pca.fit_transform(VC_df)
    newdf2 = pd.DataFrame(newdata2)
    newdf2.to_csv('./Data/Data for svr/VC_CPA.csv')

    # get a .csv file for Max Delta Volume after PCA
    pca = PCA(n_components=134)
    newdata3=pca.fit_transform(MDV_df)
    newdf3 = pd.DataFrame(newdata3)
    newdf3.to_csv('./Data/Data for svr/MDV_CPA.csv')
    return


