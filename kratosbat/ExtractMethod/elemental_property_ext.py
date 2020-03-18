def get_elementProperty(clean_battery_df=None):

        # here we import a API called The Materials Agnostic Platform for Informatics and Exploration (Magpie)
        # this API can let us use formula of a conpound to get its  elemental properties from statistics of
        # atomic constituents attributes.

        #  Details are in this paper:
        #  Ward, L.; Agrawal, A.; Choudhary, A.; Wolverton, C. A General-Purpose Machine Learning Framework for Predicting Properties of 
        #  Inorganic Materials. npj Comput. Mater. 2016, 2, No. 16028.
        from magpie import MagpieServer
        import os
        import pandas as pd
        m = MagpieServer()
        if clean_battery_df == None:
            if os.path.exists('../Data/BatteryData.csv'):
                clean_battery_df=pd.read_csv('../Data/BatteryData.csv')
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
        element_attributes.to_csv(path_or_buf='../Data/ElementalProperty.csv')
        return element_attributes

